# -*- coding: utf-8 -*-
"""
Title   : 외부데이터 기반 LSTM 예측 및 멀티 실험 자동화 모듈 (예측값 JSON 기록 기능 추가)
Author  : 주성중 / (주)맵인어스
Description : 태양광 발전량 예측을 위한 LSTM 모델 학습 및 실험 자동화 시스템
             - PostgreSQL DB에서 태양광 발전 데이터 로드
             - 다중 실험 설정을 JSON 파일로 관리
             - LSTM 모델 학습/예측 및 성능 평가
             - 예측 결과를 JSON/그래프로 저장
             - 실험 결과를 DB에 자동 저장
"""

import os
# TensorFlow 최적화 옵션 비활성화 (경고 메시지 감소)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# TensorFlow 로그 레벨 설정 (2 = ERROR만 출력)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
import psycopg2
from sklearn.preprocessing import StandardScaler
import json
import joblib
from sqlalchemy import create_engine
from datetime import datetime

# ============================================================================
# 환경 설정 및 경로 설정
# ============================================================================

# 실행 환경 확인 (local 또는 production)
ENV = os.getenv('FLASK_ENV', 'local')
if ENV == 'local':
    # 로컬 개발 환경 경로
    root = "D:/work/lstm"
else:
    # 프로덕션 환경 경로
    root = "/app/webfiles/lstm"

# 그래프 이미지 저장 경로 (학습 손실, 예측 결과 그래프 등)
graph_path = os.path.abspath(root + "/graphImage")
os.makedirs(graph_path, exist_ok=True)

# 학습된 모델 파일 저장 경로 (.h5, .json, .pkl)
model_path = os.path.abspath(root + "/saved_models")
os.makedirs(model_path, exist_ok=True)

# 예측 결과 JSON 파일 저장 경로
prediction_path = os.path.abspath(root + "/predictions")
os.makedirs(prediction_path, exist_ok=True)

# ============================================================================
# 데이터베이스 연결 함수
# ============================================================================

def get_db_engine():
    """
    PostgreSQL 데이터베이스 연결을 위한 SQLAlchemy 엔진 생성
    
    Returns:
        Engine: SQLAlchemy 데이터베이스 엔진 객체
    """
    # 데이터베이스 연결 문자열 (실제 운영시 환경변수로 관리 권장)
    connection_string = "postgresql://postgres:mapinus@10.10.10.201:5432/postgres"
    # connection_string = "postgresql://postgres:mapinus%401004@10.10.10.201:5434/postgres"
    # connection_string = "postgresql://postgres:7926@localhost:5432/postgres"  # 로컬 테스트용
    return create_engine(connection_string)

# ============================================================================
# JSON 설정 파일 관리 함수
# ============================================================================

def load_experiments_config(config_file="experiments.json"):
    """
    실험 설정이 담긴 JSON 파일을 로드
    
    JSON 파일 구조 예시:
    {
        "experiments": [
            {
                "name": "태양광발전량_실험1",
                "modelName": "solar_power_model_1",
                "tablename": "solar_data",
                "dateColumn": "timestamp",
                "studyColumns": "temperature,humidity,solar_radiation,power_output",
                "targetColumn": "power_output",
                "r_epochs": 100,
                "r_batchSize": 32,
                "r_validationSplit": 0.2,
                "r_seqLen": 60,
                "r_predDays": 1
            }
        ]
    }
    
    Args:
        config_file (str): 설정 파일 경로 (기본값: experiments.json)
        
    Returns:
        list: 실험 설정 딕셔너리 리스트
    """
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config['experiments']
    except FileNotFoundError:
        print(f"❌ 설정 파일 '{config_file}'을 찾을 수 없습니다.")
        return []
    except json.JSONDecodeError:
        print(f"❌ JSON 파일 형식이 잘못되었습니다: {config_file}")
        return []

# ============================================================================
# 데이터 로드 함수
# ============================================================================

def load_data_from_db(tablename, dateColumn, studyColumns, cust_id):
    """
    PostgreSQL 데이터베이스에서 태양광 발전 데이터를 로드
    
    특징:
    - 특정 날짜 제외 (장비 점검일, 이상 데이터 발생일 등)
    - 시간대별 데이터 분포 확인 및 출력
    - 시간순 정렬
    
    Args:
        tablename (str): 데이터베이스 테이블명 (예: 'solar_power_data')
        dateColumn (str): 날짜/시간 컬럼명 (예: 'timestamp', 'datetime')
        studyColumns (str): 학습에 사용할 컬럼들 (쉼표로 구분, 예: 'temp,humidity,radiation,power')
        
    Returns:
        DataFrame: 로드된 데이터프레임 또는 None (오류시)
    """
    try:
        engine = get_db_engine()
        
        # SQL 쿼리 작성
        # - studyColumns와 dateColumn을 선택
        # - dateColumn이 NULL이 아닌 데이터만
        # - 특정 날짜들을 제외 (장비 점검일, 이상 데이터 발생일 등)
        # - 시간순 정렬
        query = f"""
        SELECT {studyColumns},{dateColumn}
        FROM carbontwin.{tablename}
        WHERE {dateColumn} IS NOT NULL
        AND TO_CHAR({dateColumn}, 'MM-DD') NOT IN (
            '06-02', '06-13', '06-14', '06-15', '06-16', '06-17',
            '06-20', '06-21', '06-24', '06-25', '06-26', '06-28',
            '07-01', '07-08', '07-13', '07-14', '07-15', '07-16',
            '07-17', '07-18', '07-19', '07-21', '07-22', '11-02'
        )
        AND cust_id = {cust_id}
        ORDER BY {dateColumn} ASC
        """
        print("cust_id : ",cust_id);
        
        # SQL 쿼리 실행 및 데이터프레임 생성
        data = pd.read_sql_query(query, engine)

        # 시간대 분포 확인 및 출력 (데이터 품질 체크)
        if dateColumn in data.columns and len(data) > 0:
            # 날짜 컬럼을 datetime 형식으로 변환
            data[dateColumn] = pd.to_datetime(data[dateColumn])
            # 시간(hour) 추출
            hours = data[dateColumn].dt.hour
            print(f"   📊 시간 범위: {hours.min()}시 ~ {hours.max()}시")
            # 시간대별 데이터 개수 계산
            hour_counts = hours.value_counts().sort_index()
            print(f"   📊 시간대별 데이터 수:")
            for hour, count in hour_counts.items():
                print(f"      {hour:2d}시: {count:5d}개")
        
        return data
        
    except Exception as e:
        print(f"❌ 데이터베이스 오류: {str(e)}")
        return None

# ============================================================================
# JSON 직렬화 변환 함수
# ============================================================================

def convert_numpy_to_json_serializable(obj):
    """
    NumPy 배열과 특수 타입을 JSON 직렬화 가능한 형태로 변환
    
    NumPy 배열과 pandas의 특수 타입은 JSON으로 직접 저장할 수 없으므로
    Python 기본 타입으로 변환이 필요
    
    Args:
        obj: 변환할 객체 (numpy array, numpy scalar, pandas Timestamp 등)
        
    Returns:
        JSON 직렬화 가능한 Python 기본 타입 (list, int, float, str)
    """
    if isinstance(obj, np.ndarray):
        # NumPy 배열 → Python 리스트
        return obj.tolist()
    elif isinstance(obj, np.integer):
        # NumPy 정수 → Python int
        return int(obj)
    elif isinstance(obj, np.floating):
        # NumPy 실수 → Python float
        return float(obj)
    elif isinstance(obj, pd.Timestamp):
        # pandas Timestamp → ISO 형식 문자열
        return obj.isoformat()
    elif isinstance(obj, datetime):
        # datetime → ISO 형식 문자열
        return obj.isoformat()
    else:
        return obj

# ============================================================================
# 예측 결과 JSON 저장 함수
# ============================================================================

def save_predictions_to_json(modelName, dates, actual_values, predicted_values, target_column):
    """
    LSTM 모델의 예측 결과를 JSON 파일로 저장
    
    저장 내용:
    - 각 시점별 실제값과 예측값
    - 오차 및 오차율
    - 통계 정보 (최소/최대/평균, MAE, RMSE)
    
    Args:
        modelName (str): 모델 이름 (파일명으로 사용)
        dates: 예측 시점의 날짜/시간 배열
        actual_values (array): 실제 태양광 발전량 값
        predicted_values (array): 모델이 예측한 발전량 값
        target_column (str): 예측 대상 컬럼명 (예: 'power_output')
        
    Returns:
        dict: 예측 결과 요약 딕셔너리 또는 None (오류시)
    """
    try:
        # 각 시점별 예측 데이터 구성
        predictions_data = []
        
        for i in range(len(actual_values)):
            prediction_record = {
                "index": i,  # 순번
                "date": convert_numpy_to_json_serializable(dates.iloc[i] if hasattr(dates, 'iloc') else dates[i]),
                "actual_value": convert_numpy_to_json_serializable(actual_values[i]),  # 실제 발전량
                "predicted_value": convert_numpy_to_json_serializable(predicted_values[i]),  # 예측 발전량
                "difference": convert_numpy_to_json_serializable(predicted_values[i] - actual_values[i]),  # 차이값
                "percentage_error": convert_numpy_to_json_serializable(
                    abs((predicted_values[i] - actual_values[i]) / actual_values[i] * 100) if actual_values[i] != 0 else 0
                )  # 백분율 오차
            }
            predictions_data.append(prediction_record)
        
        # JSON 파일 저장 경로
        prediction_file_path = os.path.join(prediction_path, f"{modelName}_predictions.json")
        
        # 전체 예측 결과 요약 구조
        prediction_summary = {
            "model_name": modelName,
            "target_column": target_column,
            "prediction_count": len(predictions_data),  # 총 예측 개수
            "timestamp": datetime.now().isoformat(),  # 저장 시각
            "statistics": {
                # 실제값 통계
                "actual_min": convert_numpy_to_json_serializable(np.min(actual_values)),
                "actual_max": convert_numpy_to_json_serializable(np.max(actual_values)),
                "actual_mean": convert_numpy_to_json_serializable(np.mean(actual_values)),
                # 예측값 통계
                "predicted_min": convert_numpy_to_json_serializable(np.min(predicted_values)),
                "predicted_max": convert_numpy_to_json_serializable(np.max(predicted_values)),
                "predicted_mean": convert_numpy_to_json_serializable(np.mean(predicted_values)),
                # 오차 통계
                "mean_absolute_error": convert_numpy_to_json_serializable(np.mean(np.abs(predicted_values - actual_values))),
                "rmse": convert_numpy_to_json_serializable(np.sqrt(np.mean((predicted_values - actual_values) ** 2)))
            },
            "predictions": predictions_data  # 상세 예측 데이터
        }
        
        # JSON 파일로 저장 (한글 깨짐 방지, 들여쓰기 적용)
        with open(prediction_file_path, 'w', encoding='utf-8') as f:
            json.dump(prediction_summary, f, indent=2, ensure_ascii=False)
        
        print(f"💾 예측 결과가 저장되었습니다: {prediction_file_path}")
        return prediction_summary
        
    except Exception as e:
        print(f"❌ 예측 결과 저장 중 오류: {str(e)}")
        return None

# ============================================================================
# 실험 결과 DB 저장 함수
# ============================================================================

def save_experiment_to_db(result, config, is_new_model):
    """
    실험 결과를 PostgreSQL 데이터베이스에 저장
    
    두 개의 테이블에 저장:
    1. lstm_model: 모델 기본 정보 (신규 모델인 경우에만)
    2. lstm_experiment: 각 실험별 성능 지표 및 결과 파일 경로
    
    Args:
        result (dict): 실험 결과 딕셔너리 (정확도, MAPE, 그래프 경로 등)
        config (dict): 실험 설정 딕셔너리
        is_new_model (bool): 신규 모델 여부 (True: 최초 생성, False: 재학습)
        
    Returns:
        bool: 저장 성공 여부
    """
    try:
        engine = get_db_engine()
        model_name = result.get('modelName')
        
        # 신규 모델인 경우 lstm_model 테이블에 등록
        if is_new_model:
            # 이미 존재하는 모델인지 확인
            check_query = f"SELECT model_id FROM carbontwin.lstm_model WHERE model_name = '{model_name}'"
            existing = pd.read_sql_query(check_query, engine)
            
            if existing.empty:
                # 모델 기본 정보 딕셔너리 생성
                model_data = {
                    'model_name': model_name,
                    'target_column': config.get('targetColumn'),  # 예측 대상 (예: 발전량)
                    'date_column': config.get('dateColumn'),
                    'study_columns': config.get('studyColumns'),  # 입력 변수들
                    'epochs': config.get('r_epochs'),  # 학습 반복 횟수
                    'batch_size': config.get('r_batchSize'),  # 배치 크기
                    'validation_split': config.get('r_validationSplit'),  # 검증 데이터 비율
                    'sequence_length': config.get('r_seqLen'),  # 시퀀스 길이 (과거 몇 개 데이터 사용)
                    'prediction_days': config.get('r_predDays'),  # 예측 간격
                    'created_at': datetime.now()
                }
                
                # 데이터베이스에 저장
                df_model = pd.DataFrame([model_data])
                df_model.to_sql('lstm_model', engine, schema='carbontwin',
                              if_exists='append', index=False)
                print(f"✅ 신규 모델 등록: {model_name}")
            else:
                print(f"ℹ️  기존 모델 사용: {model_name}")
        
        # model_id 조회 (외래키로 사용)
        query = f"SELECT model_id FROM carbontwin.lstm_model WHERE model_name = '{model_name}'"
        model_id = pd.read_sql_query(query, engine).iloc[0]['model_id']
        
        # 실험 결과 정보 딕셔너리 생성
        experiment_data = {
            'model_id': model_id,  # 외래키
            'experiment_name': result.get('experiment_name', config.get('name')),
            'accuracy': result.get('accuracy'),  # 정확도 (100 - MAPE)
            'mape': result.get('mape'),  # 평균 절대 백분율 오차
            'rmse': result.get('rmse'),  # 평균 제곱근 오차
            'r2_score': result.get('r2_score'),  # 결정계수
            'model_file_path': os.path.abspath(os.path.join(model_path, f"{model_name}.h5")),  # 모델 파일 경로
            'training_loss_img_path': os.path.abspath(os.path.join(root, result.get('training_loss_img'))),  # 학습 손실 그래프
            'total_graph_img_path': os.path.abspath(os.path.join(root, result.get('total_graph_img'))),  # 전체 예측 그래프
            'diff_graph_img_path': os.path.abspath(os.path.join(root, result.get('diff_graph_img'))),  # 확대 예측 그래프
            'prediction_file_path': os.path.abspath(os.path.join(root, result.get('prediction_file'))),  # 예측 JSON 파일
            'execution_time_seconds': result.get('execution_time'),  # 실행 시간
            'status': result.get('status'),  # 실험 상태 (success/error)
            'config_json': json.dumps(config, ensure_ascii=False),  # 설정 전체를 JSON 문자열로 저장
            'created_at': datetime.now()
        }
        
        # 데이터베이스에 저장
        df_experiment = pd.DataFrame([experiment_data])
        df_experiment.to_sql('lstm_experiment', engine, schema='carbontwin',
                           if_exists='append', index=False)
        
        print(f"💾 실험 결과 저장 완료 (Model ID: {model_id})")
        return True
        
    except Exception as e:
        print(f"❌ DB 저장 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# 단일 실험 실행 함수
# ============================================================================

def run_single_experiment(experiment_config, experiment_index):
    """
    단일 실험을 실행하고 결과를 DB에 저장
    
    프로세스:
    1. 데이터베이스에서 데이터 로드
    2. LSTM 모델 학습/예측 수행
    3. 실행 시간 측정
    4. 결과를 DB에 저장
    
    Args:
        experiment_config (dict): 실험 설정 딕셔너리
        experiment_index (int): 실험 순번 (0부터 시작)
        
    Returns:
        dict: 실험 결과 딕셔너리
    """
    print(f"\n{'='*60}")
    print(f"🚀 실험 {experiment_index + 1} 시작: {experiment_config['name']}")
    print(f"{'='*60}")
    
    # 1. 데이터 로드
    data = load_data_from_db(
        experiment_config['tablename'],
        experiment_config['dateColumn'], 
        experiment_config['studyColumns'],
        experiment_config['cust_id']
    )
    
    if data is None:
        return {"status": "error", "message": "데이터 로드 실패"}
    
    # 2. LSTM 학습 및 예측 실행
    start_time = time.time()
    result = lstmFinance(data, experiment_config)
    end_time = time.time()
    
    # 실행 시간 기록
    result['execution_time'] = round(end_time - start_time, 2)
    result['experiment_name'] = experiment_config['name']
    
    print(f"⏱️  실험 완료 시간: {result['execution_time']}초")
    
    # 3. 실험 결과를 DB에 저장
    if result['status'] == 'success':
        print(f"\n💾 데이터베이스에 결과 저장 중...")
        save_success = save_experiment_to_db(
            result, 
            experiment_config,
            is_new_model=result.get('is_new_model', False)
        )
        
        if save_success:
            print(f"✅ 데이터베이스 저장 완료")
    
    return result

# ============================================================================
# LSTM 모델 학습 및 예측 함수 (핵심 함수)
# ============================================================================

def lstmFinance(lstmData, config):
    """
    태양광 발전량 예측을 위한 LSTM 모델 학습 및 예측 수행
    
    주요 단계:
    1. 데이터 전처리 및 검증
    2. 데이터 정규화 (StandardScaler)
    3. 시퀀스 데이터 생성 (시계열 → LSTM 입력 형태)
    4. 모델 학습 또는 기존 모델 로드
    5. 예측 수행
    6. 성능 평가 (MAPE, RMSE, R², 방향성 정확도 등)
    7. 결과 시각화 및 저장
    
    Args:
        lstmData (DataFrame): 학습/예측에 사용할 데이터프레임
        config (dict): 실험 설정 (모델명, 학습 파라미터 등)
        
    Returns:
        dict: 실험 결과 딕셔너리 (정확도, 그래프 경로, 예측 데이터 등)
    """
    
    # TensorFlow eager execution 활성화 (디버깅 용이)
    if not tf.executing_eagerly():
        tf.config.run_functions_eagerly(True)

    # ========================================================================
    # 1. 설정 파라미터 추출
    # ========================================================================
    modelName = config['modelName']  # 모델 이름
    dateColumn = config['dateColumn']  # 날짜 컬럼명
    studyColumns = config['studyColumns']  # 학습에 사용할 컬럼들 (쉼표 구분)
    targetColumn = config['targetColumn']  # 예측 대상 컬럼 (예: 발전량)
    r_epochs = config['r_epochs']  # 학습 에포크 수
    r_batchSize = config['r_batchSize']  # 배치 크기
    r_validationSplit = config['r_validationSplit']  # 검증 데이터 비율
    r_seqLen = config['r_seqLen']  # 시퀀스 길이 (과거 몇 개 시점 데이터 사용)
    r_predDays = config['r_predDays']  # 예측 간격 (몇 시점 후 예측)

    # 파일 경로 설정
    training_loss_path = os.path.join(graph_path, f"{modelName}_trainingLoss.png")  # 학습 손실 그래프
    total_graph_path = os.path.join(graph_path, f"{modelName}_totalgraph.png")  # 전체 예측 그래프
    diff_graph_path = os.path.join(graph_path, f"{modelName}_diffgraph.png")  # 확대 예측 그래프
    model_file_path = os.path.join(model_path, f"{modelName}.h5")  # 모델 저장 파일

    stock_data = lstmData
    
    # ========================================================================
    # 2. 데이터 검증
    # ========================================================================
    if stock_data.empty:
        return {"status": "error", "message": "데이터가 비어있습니다."}
    
    print(f"\n📊 로드된 데이터 정보:")
    print(f"   - 총 데이터 수: {len(stock_data)}개")
    
    # 학습 컬럼 리스트 생성
    study_columns_list = [col.strip() for col in studyColumns.split(',')]
    
    # 타겟 컬럼이 학습 컬럼에 포함되어 있는지 확인
    if targetColumn not in study_columns_list:
        return {"status": "error", "message": f"타겟 컬럼 '{targetColumn}'이 학습 컬럼에 없습니다."}

    # ========================================================================
    # 3. 날짜 컬럼 처리 및 시간 범위 확인
    # ========================================================================
    if dateColumn in stock_data.columns:
        # 날짜 컬럼을 datetime 형식으로 변환
        dates = pd.to_datetime(stock_data[dateColumn], errors='coerce')
        
        # 시간 범위 확인 (태양광 발전은 주로 낮 시간대)
        hours = dates.dt.hour
        print(f"   - 시간 범위: {hours.min()}시 ~ {hours.max()}시")
        print(f"   - 고유 시간대: {sorted(hours.unique())}")
    else:
        # 날짜 컬럼이 없으면 가상 날짜 생성 (15분 간격)
        dates = pd.date_range(start='2023-01-01', periods=len(stock_data), freq='15T')
        print(f"⚠️ 날짜 컬럼 '{dateColumn}'이 없어서 가상 날짜를 생성했습니다.")
    
    # 원본 타겟값 저장 (나중에 그래프에 사용)
    original_open = stock_data[targetColumn].values
    
    # 학습용 데이터만 추출 (모든 study_columns)
    stock_data_for_training = stock_data[study_columns_list].astype(float)

    # ========================================================================
    # 4. 데이터 정규화 (StandardScaler)
    # ========================================================================
    # StandardScaler: 평균 0, 표준편차 1로 정규화
    # LSTM은 정규화된 데이터에서 더 잘 학습됨
    scaler = StandardScaler()
    stock_data_scaled = scaler.fit_transform(stock_data_for_training)

    # ========================================================================
    # 5. 학습/테스트 데이터 분할 (80% 학습, 20% 테스트)
    # ========================================================================
    split_index = int(len(stock_data_scaled) * 0.8)
    train_data_scaled = stock_data_scaled[:split_index]  # 처음 80%
    test_data_scaled = stock_data_scaled[split_index:]  # 나머지 20%
    test_dates = dates[split_index:]  # 테스트 데이터의 날짜

    # ========================================================================
    # 6. 시퀀스 데이터 생성 파라미터 설정
    # ========================================================================
    pred_days = int(r_predDays)  # 예측 간격
    seq_len = int(r_seqLen)  # 시퀀스 길이 (예: 60 → 과거 60개 시점 데이터 사용)
    input_dim = stock_data_for_training.shape[1]  # 입력 변수 개수
    target_idx = study_columns_list.index(targetColumn)  # 타겟 컬럼의 인덱스

    # ========================================================================
    # 7. 데이터 충분성 검증
    # ========================================================================
    print(f"\n🔍 시퀀스 생성 검증:")
    print(f"   - 전체 데이터: {len(stock_data_scaled)}개")
    print(f"   - 학습 데이터: {len(train_data_scaled)}개")
    print(f"   - 테스트 데이터: {len(test_data_scaled)}개")
    print(f"   - 시퀀스 길이(seq_len): {seq_len}")
    print(f"   - 예측 간격(pred_days): {pred_days}")
    
    # 최소 필요 데이터 수 = 시퀀스 길이 + 예측 간격
    min_required = seq_len + pred_days
    print(f"   - 필요한 최소 데이터: {min_required}개")
    
    # 학습 데이터 충분성 확인
    if len(train_data_scaled) < min_required:
        error_msg = f"학습 데이터 부족: {len(train_data_scaled)}개 (최소 {min_required}개 필요)"
        print(f"❌ {error_msg}")
        return {"status": "error", "message": error_msg}
    
    # 테스트 데이터 충분성 확인
    if len(test_data_scaled) < min_required:
        error_msg = f"테스트 데이터 부족: {len(test_data_scaled)}개 (최소 {min_required}개 필요)"
        print(f"❌ {error_msg}")
        return {"status": "error", "message": error_msg}

    # ========================================================================
    # 8. 시퀀스 데이터 생성
    # ========================================================================
    # LSTM 입력 형태: (samples, timesteps, features)
    # - samples: 시퀀스 개수
    # - timesteps: 시퀀스 길이 (seq_len)
    # - features: 입력 변수 개수 (input_dim)
    
    trainX, trainY, testX, testY = [], [], [], []
    
    # 학습 데이터 시퀀스 생성 범위
    # seq_len부터 시작 (그 이전은 시퀀스를 만들 수 없음)
    # len - pred_days + 1까지 (그 이후는 예측 타겟이 없음)
    train_range = range(seq_len, len(train_data_scaled) - pred_days + 1)
    test_range = range(seq_len, len(test_data_scaled) - pred_days + 1)
    
    print(f"\n📊 시퀀스 생성 범위:")
    print(f"   - 학습 시퀀스: {len(train_range)}개")
    print(f"   - 테스트 시퀀스: {len(test_range)}개")
    
    # 시퀀스 생성 가능 여부 확인
    if len(train_range) == 0:
        return {"status": "error", "message": "학습 시퀀스를 생성할 수 없습니다."}
    
    if len(test_range) == 0:
        return {"status": "error", "message": "테스트 시퀀스를 생성할 수 없습니다."}
    
    # 학습 데이터 시퀀스 생성
    # trainX: 과거 seq_len개 시점의 모든 변수 데이터
    # trainY: pred_days 후의 타겟 변수 값
    for i in train_range:
        trainX.append(train_data_scaled[i - seq_len:i, 0:input_dim])  # 과거 seq_len개 시점
        trainY.append(train_data_scaled[i + pred_days - 1:i + pred_days, target_idx])  # pred_days 후 타겟값

    # 테스트 데이터 시퀀스 생성 (동일한 방식)
    for i in test_range:
        testX.append(test_data_scaled[i - seq_len:i, 0:input_dim])
        testY.append(test_data_scaled[i + pred_days - 1:i + pred_days, target_idx])

    # NumPy 배열로 변환 (TensorFlow 입력 형식)
    trainX, trainY = np.array(trainX), np.array(trainY)
    testX, testY = np.array(testX), np.array(testY)

    print(f"✅ 시퀀스 생성 완료:")
    print(f"   - trainX: {trainX.shape}, trainY: {trainY.shape}")
    print(f"   - testX: {testX.shape}, testY: {testY.shape}")

    # ========================================================================
    # 9. LSTM 모델 생성 또는 로드
    # ========================================================================
    print(f"\n🔄 {modelName} 모델 학습 시작...")
    is_new_model = False

    try:
        # 기존 모델 파일이 있으면 로드
        model = load_model(model_file_path, compile=False)
        model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
        print("✅ 기존 모델 로드됨")
        is_new_model = False
        
    except (OSError, IOError):
        # 모델 파일이 없으면 새로 생성
        print("🔄 새 모델 생성 중...")
        is_new_model = True

        # LSTM 모델 구조
        # - Input: (timesteps, features) 형태
        # - LSTM(64): 첫 번째 LSTM 레이어, 64개 유닛, return_sequences=True (다음 LSTM으로 전달)
        # - LSTM(32): 두 번째 LSTM 레이어, 32개 유닛, return_sequences=False (마지막 시점만 출력)
        # - Dense: 출력 레이어, 예측값 출력
        model = Sequential([
            Input(shape=(trainX.shape[1], trainX.shape[2])),  # (seq_len, input_dim)
            LSTM(64, return_sequences=True),  # 첫 번째 LSTM 층
            LSTM(32, return_sequences=False),  # 두 번째 LSTM 층
            Dense(trainY.shape[1])  # 출력 층 (예측값)
        ])

        # 모델 컴파일
        # - optimizer: Adam (학습률 0.01)
        # - loss: MSE (평균 제곱 오차)
        model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')

        # ====================================================================
        # 10. 학습 진행 상황 출력을 위한 커스텀 콜백 클래스
        # ====================================================================
        class TrainingCallback(Callback):
            """
            학습 진행 상황을 상세하게 출력하는 커스텀 콜백
            - 에포크 시작/종료 시 진행률 바 표시
            - 손실 및 검증 손실 출력
            - 검증 손실 개선/악화 알림
            """
            def __init__(self, total_epochs, batch_size):
                super().__init__()
                self.total_epochs = total_epochs
                self.batch_size = batch_size
                self.prev_val_loss = None
                
            def on_train_begin(self, logs=None):
                """학습 시작 시"""
                print(f"🚀 모델 학습 시작 - 총 {self.total_epochs} 에포크")
                print(f"📊 배치 크기: {self.batch_size}")
                
            def on_epoch_begin(self, epoch, logs=None):
                """각 에포크 시작 시"""
                print(f"\n⏳ Epoch {epoch + 1}/{self.total_epochs} 시작...")
                
            def on_epoch_end(self, epoch, logs=None):
                """각 에포크 종료 시"""
                logs = logs or {}
                loss = logs.get('loss', 0)  # 학습 손실
                val_loss = logs.get('val_loss', 0)  # 검증 손실
                
                # 진행률 바 생성
                progress = (epoch + 1) / self.total_epochs * 100
                bar_length = 30
                filled_length = int(bar_length * (epoch + 1) // self.total_epochs)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                
                print(f"✅ Epoch {epoch + 1}/{self.total_epochs} [{bar}] {progress:.1f}%")
                print(f"   📉 Loss: {loss:.6f} | Val_Loss: {val_loss:.6f}")
                
                # 검증 손실 변화 추적
                if epoch > 0 and self.prev_val_loss is not None:
                    if val_loss < self.prev_val_loss:
                        print(f"   📈 검증 손실 개선: {self.prev_val_loss:.6f} → {val_loss:.6f}")
                    elif val_loss > self.prev_val_loss * 1.1:  # 10% 이상 증가
                        print(f"   ⚠️  검증 손실 증가: {self.prev_val_loss:.6f} → {val_loss:.6f}")
                
                self.prev_val_loss = val_loss
                
            def on_train_end(self, logs=None):
                """학습 완료 시"""
                print(f"\n🎉 학습 완료!")

        # ====================================================================
        # 11. 모델 학습 수행
        # ====================================================================
        history = model.fit(
            trainX, trainY,
            epochs=int(r_epochs),  # 학습 에포크 수
            batch_size=int(r_batchSize),  # 배치 크기
            validation_split=float(r_validationSplit),  # 검증 데이터 비율 (예: 0.2 = 20%)
            verbose=1,  # 학습 진행 상황 출력
            callbacks=[TrainingCallback(int(r_epochs), int(r_batchSize))]  # 커스텀 콜백 적용
        )

        # 학습된 모델 저장 (.h5 형식)
        model.save(model_file_path)
        print("✅ 모델 저장 완료")

        # ====================================================================
        # 12. 학습 손실 그래프 생성 및 저장
        # ====================================================================
        plt.figure(figsize=(12, 4))
        plt.plot(history.history['loss'], label='Training loss')  # 학습 손실
        plt.plot(history.history['val_loss'], label='Validation loss')  # 검증 손실
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{modelName} - Training Loss')
        plt.legend()
        plt.savefig(training_loss_path)
        plt.close()

    # ========================================================================
    # 13. 예측 수행
    # ========================================================================
    print(f"\n🔮 예측 수행 중...")
    print(f"📊 예측할 샘플 수: {len(testX)}")
    
    # 배치 단위로 예측 (메모리 효율성)
    batch_size_pred = 32
    predictions = []
    total_batches = (len(testX) + batch_size_pred - 1) // batch_size_pred
    
    # 진행률 표시하며 배치별 예측 수행
    for i in range(0, len(testX), batch_size_pred):
        batch_end = min(i + batch_size_pred, len(testX))
        batch_data = testX[i:batch_end]
        
        # 예측 수행 (verbose=0: 출력 없음)
        batch_pred = model.predict(batch_data, verbose=0)
        predictions.append(batch_pred)
        
        # 진행률 바 출력
        current_batch = (i // batch_size_pred) + 1
        progress = current_batch / total_batches * 100
        bar_length = 25
        filled_length = int(bar_length * current_batch // total_batches)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        print(f"\r⏳ 예측 진행: [{bar}] {progress:.1f}% ({current_batch}/{total_batches} 배치)", end='', flush=True)
    
    # 배치별 예측 결과를 하나로 합침
    prediction = np.vstack(predictions)
    print(f"\n✅ 예측 완료! 총 {len(prediction)}개 샘플 예측됨")

    # ========================================================================
    # 14. 예측 결과 역정규화 (원래 스케일로 복원)
    # ========================================================================
    # StandardScaler로 정규화했으므로, 원래 값으로 되돌려야 함
    
    # 예측값 역변환
    # mean_values_pred: 모든 변수의 평균값으로 채운 배열 생성
    mean_values_pred = np.repeat(scaler.mean_[np.newaxis, :], prediction.shape[0], axis=0)
    mean_values_pred[:, target_idx] = np.squeeze(prediction)  # 타겟 컬럼만 예측값으로 대체
    y_pred = scaler.inverse_transform(mean_values_pred)[:, target_idx]  # 역변환 후 타겟 컬럼 추출

    # 실제값 역변환 (동일한 방식)
    mean_values_testY = np.repeat(scaler.mean_[np.newaxis, :], testY.shape[0], axis=0)
    mean_values_testY[:, target_idx] = np.squeeze(testY)
    testY_original = scaler.inverse_transform(mean_values_testY)[:, target_idx]
    
    # 예측에 해당하는 날짜 추출
    valid_test_dates = test_dates[seq_len : seq_len + len(testY_original)]

    # ========================================================================
    # 15. 예측 결과를 JSON 파일로 저장
    # ========================================================================
    print(f"\n💾 예측 결과를 JSON 파일로 저장 중...")
    prediction_summary = save_predictions_to_json(
        modelName, 
        valid_test_dates, 
        testY_original, 
        y_pred, 
        targetColumn
    )

    # ========================================================================
    # 16. 전체 예측 결과 그래프 생성 및 저장
    # ========================================================================
    plt.figure(figsize=(15, 5))
    # 전체 원본 데이터 (초록색)
    plt.plot(dates, original_open, color='green', label=f'Original {targetColumn}', alpha=0.7)
    # 테스트 데이터 실제값 (파란색)
    plt.plot(valid_test_dates, testY_original, color='blue', label=f'Actual {targetColumn}')
    # 예측값 (빨간색 점선)
    plt.plot(valid_test_dates, y_pred, color='red', linestyle='--', label=f'Predicted {targetColumn}')
    plt.xlabel(dateColumn)
    plt.ylabel(f'{targetColumn} Value')
    plt.title(f'{modelName} - Prediction Results')
    plt.legend()
    plt.savefig(total_graph_path)
    plt.close()

    # ========================================================================
    # 17. 최근 50개 데이터 확대 그래프 생성 및 저장
    # ========================================================================
    zoom_start = max(0, len(valid_test_dates) - 50)
    plt.figure(figsize=(15, 5))
    # 실제값과 예측값만 표시 (최근 50개)
    plt.plot(valid_test_dates[zoom_start:], testY_original[zoom_start:], color='blue', label=f'Actual {targetColumn}')
    plt.plot(valid_test_dates[zoom_start:], y_pred[zoom_start:], color='red', linestyle='--', label=f'Predicted {targetColumn}')
    plt.xlabel(dateColumn)
    plt.ylabel(f'{targetColumn} Value')
    plt.title(f'{modelName} - Recent Predictions (Last 50 points)')
    plt.legend()
    plt.savefig(diff_graph_path)
    plt.close()

    # ========================================================================
    # 18. 성능 평가 지표 계산
    # ========================================================================
    print(f"\n📈 성능 평가 중...")
    
    # ====================================================================
    # MAPE (Mean Absolute Percentage Error) 계산 함수
    # ====================================================================
    def mean_absolute_percentage_error(y_true, y_pred, valid_test_dates):
        """
        평균 절대 백분율 오차 계산
        
        특징:
        - 임계값(eps) 이상의 값만 계산에 포함 (너무 작은 값 제외)
        - 모든 예측 데이터를 상세하게 출력
        - 오차 통계 제공
        
        Args:
            y_true: 실제값 배열
            y_pred: 예측값 배열
            valid_test_dates: 날짜 배열
            
        Returns:
            float: MAPE 값 (백분율)
        """
        eps = 9  # 임계값 (태양광 발전량이 9 이하인 경우 제외)
        mask = y_true > eps  # 임계값 초과 데이터만 선택
        
        print(f"\n📊 MAPE 계산 정보:")
        print(f"   - 임계값(eps): {eps}")
        print(f"   - 전체 데이터: {len(y_true)}개")
        print(f"   - 임계값 초과 데이터: {np.sum(mask)}개")
        
        if np.sum(mask) == 0:
            print("   ⚠️ 임계값을 초과하는 데이터가 없습니다.")
            return 999.0
        
        # 임계값 초과하는 데이터 필터링
        filtered_dates = valid_test_dates[mask]
        filtered_true = y_true[mask]
        filtered_pred = y_pred[mask]
        
        # 전체 예측 데이터 상세 출력
        print(f"\n📋 임계값 초과 데이터 전체 ({len(filtered_true)}개):")
        print(f"{'='*90}")
        print(f"{'날짜/시간':<25} {'실제값':>12} {'예측값':>12} {'오차':>12} {'오차율(%)':>12}")
        print(f"{'-'*90}")
        
        for i in range(len(filtered_true)):
            date_str = filtered_dates.iloc[i].strftime('%Y-%m-%d %H:%M:%S') if hasattr(filtered_dates.iloc[i], 'strftime') else str(filtered_dates.iloc[i])
            true_val = filtered_true[i]
            pred_val = filtered_pred[i]
            error = pred_val - true_val
            error_pct = abs(error / true_val * 100)
            
            print(f"{date_str:<25} {true_val:>12.4f} {pred_val:>12.4f} {error:>12.4f} {error_pct:>12.2f}")
        
        print(f"{'='*90}")
        
        # MAPE 계산 공식: 평균(|예측값 - 실제값| / 실제값) × 100
        mape_value = np.mean(np.abs((filtered_pred - filtered_true) / filtered_true)) * 100
        print(f"\n   ✅ 계산된 MAPE: {mape_value:.2f}%")
        
        # 추가 오차 통계
        errors = filtered_pred - filtered_true
        print(f"\n📊 오차 분석:")
        print(f"   - 평균 오차: {np.mean(errors):.4f}")
        print(f"   - 오차 표준편차: {np.std(errors):.4f}")
        print(f"   - 최대 과대예측: {np.max(errors):.4f}")
        print(f"   - 최대 과소예측: {np.min(errors):.4f}")
        print(f"   - 과대예측 비율: {np.sum(errors > 0) / len(errors) * 100:.1f}%")
        print(f"   - 과소예측 비율: {np.sum(errors < 0) / len(errors) * 100:.1f}%")
        
        return mape_value

    # scikit-learn 사용 가능 여부 확인
    try:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        sklearn_available = True
    except ImportError:
        print("⚠️ scikit-learn이 설치되지 않았습니다. 기본 지표만 계산합니다.")
        sklearn_available = False
    
    # MAPE 및 정확도 계산
    mape = mean_absolute_percentage_error(testY_original, y_pred, valid_test_dates)
    accuracy = 100 - mape if not np.isnan(mape) else 0  # 정확도 = 100 - MAPE
    
    # 추가 성능 지표 계산
    if sklearn_available:
        # scikit-learn 함수 사용
        mse = mean_squared_error(testY_original, y_pred)  # 평균 제곱 오차
        rmse = np.sqrt(mse)  # 평균 제곱근 오차
        mae = mean_absolute_error(testY_original, y_pred)  # 평균 절대 오차
        r2 = r2_score(testY_original, y_pred)  # 결정계수 (R²)
    else:
        # 수동으로 계산
        mse = np.mean((testY_original - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(testY_original - y_pred))
        
        # R² 수동 계산: 1 - (SS_res / SS_tot)
        ss_res = np.sum((testY_original - y_pred) ** 2)  # 잔차 제곱합
        ss_tot = np.sum((testY_original - np.mean(testY_original)) ** 2)  # 총 제곱합
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # 방향성 정확도 계산 (상승/하락 방향 예측 정확도)
    if len(testY_original) > 1:
        actual_direction = np.diff(testY_original) > 0  # 실제 상승 여부
        pred_direction = np.diff(y_pred) > 0  # 예측 상승 여부
        direction_accuracy = np.mean(actual_direction == pred_direction) * 100
    else:
        direction_accuracy = 0
    
    # ====================================================================
    # 전체 예측 결과 상세 출력 (모든 데이터)
    # ====================================================================
    print(f"\n📋 전체 예측 결과 ({len(testY_original)}개):")
    print(f"{'='*90}")
    print(f"{'날짜/시간':<25} {'실제값':>12} {'예측값':>12} {'오차':>12} {'오차율(%)':>12}")
    print(f"{'-'*90}")
    
    for i in range(len(testY_original)):
        date_str = valid_test_dates.iloc[i].strftime('%Y-%m-%d %H:%M:%S') if hasattr(valid_test_dates.iloc[i], 'strftime') else str(valid_test_dates.iloc[i])
        true_val = testY_original[i]
        pred_val = y_pred[i]
        error = pred_val - true_val
        error_pct = abs(error / true_val * 100) if true_val != 0 else 0
        
        print(f"{date_str:<25} {true_val:>12.4f} {pred_val:>12.4f} {error:>12.4f} {error_pct:>12.2f}")
    
    print(f"{'='*90}")
    
    # ====================================================================
    # 성능 결과 요약 출력
    # ====================================================================
    print(f"\n📊 모델 성능 결과:")
    print(f"   🎯 MAPE: {mape:.2f}%")
    print(f"   📈 정확도: {accuracy:.2f}%")
    print(f"   📏 MAE: {mae:.4f}")
    print(f"   📐 RMSE: {rmse:.4f}")
    print(f"   🔍 R² Score: {r2:.4f}")
    print(f"   🧭 방향성 정확도: {direction_accuracy:.2f}%")
    
    # 성능 등급 판정
    if accuracy >= 90:
        grade = "🏆 우수"
    elif accuracy >= 80:
        grade = "🥇 양호"
    elif accuracy >= 70:
        grade = "🥈 보통"
    elif accuracy >= 60:
        grade = "🥉 개선필요"
    else:
        grade = "❌ 불량"
    
    print(f"   📊 성능 등급: {grade}")
    
    # ====================================================================
    # 예측 범위 및 분포 분석
    # ====================================================================
    pred_min, pred_max = np.min(y_pred), np.max(y_pred)
    actual_min, actual_max = np.min(testY_original), np.max(testY_original)
    print(f"\n📊 예측값 범위 분석:")
    print(f"   실제값 범위: {actual_min:.3f} ~ {actual_max:.3f}")
    print(f"   예측값 범위: {pred_min:.3f} ~ {pred_max:.3f}")
    
    # 과대/과소 예측 비율
    over_predict = np.sum(y_pred > testY_original) / len(y_pred) * 100
    under_predict = 100 - over_predict
    print(f"   과예측 비율: {over_predict:.1f}%")
    print(f"   소예측 비율: {under_predict:.1f}%")

    # ========================================================================
    # 19. 설정 및 스케일러 저장
    # ========================================================================
    # 설정 파일 저장 (나중에 모델 재사용 시 필요)
    with open(os.path.join(model_path, f"{modelName}_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    # 스케일러 저장 (예측 시 동일한 정규화 적용 필요)
    joblib.dump(scaler, os.path.join(model_path, f"{modelName}_scaler.pkl"))

    # ========================================================================
    # 20. 결과 딕셔너리 구성 및 반환
    # ========================================================================
    result = {
        "status": "success",
        "modelName": modelName,
        # 생성된 파일 경로들 (상대 경로)
        "training_loss_img": f"graphImage/{modelName}_trainingLoss.png",
        "total_graph_img": f"graphImage/{modelName}_totalgraph.png",
        "diff_graph_img": f"graphImage/{modelName}_diffgraph.png",
        # 성능 지표들
        "mape": round(mape, 2),
        "accuracy": round(accuracy, 2),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2_score": round(r2, 4),
        "direction_accuracy": round(direction_accuracy, 2),
        # 예측 파일 경로
        "prediction_file": f"predictions/{modelName}_predictions.json",
        # 예측 요약 정보
        "prediction_summary": {
            "total_predictions": len(y_pred),
            "prediction_period": {
                "start_date": convert_numpy_to_json_serializable(valid_test_dates.iloc[0]) if len(valid_test_dates) > 0 else None,
                "end_date": convert_numpy_to_json_serializable(valid_test_dates.iloc[-1]) if len(valid_test_dates) > 0 else None
            },
            "value_statistics": {
                "actual_min": convert_numpy_to_json_serializable(np.min(testY_original)),
                "actual_max": convert_numpy_to_json_serializable(np.max(testY_original)),
                "actual_mean": convert_numpy_to_json_serializable(np.mean(testY_original)),
                "predicted_min": convert_numpy_to_json_serializable(np.min(y_pred)),
                "predicted_max": convert_numpy_to_json_serializable(np.max(y_pred)),
                "predicted_mean": convert_numpy_to_json_serializable(np.mean(y_pred))
            }
        }
    }
    
    # 최근 N개 예측값을 결과에 직접 포함 (빠른 참조용)
    recent_predictions_count = min(10, len(y_pred))
    if recent_predictions_count > 0:
        result["recent_predictions"] = []
        for i in range(-recent_predictions_count, 0):  # 마지막 10개
            result["recent_predictions"].append({
                "date": convert_numpy_to_json_serializable(valid_test_dates.iloc[i]),
                "actual": convert_numpy_to_json_serializable(testY_original[i]),
                "predicted": convert_numpy_to_json_serializable(y_pred[i]),
                "error": convert_numpy_to_json_serializable(abs(y_pred[i] - testY_original[i]))
            })

    # 신규 모델 여부 표시
    result['is_new_model'] = is_new_model
    return result

# ============================================================================
# 멀티 실험 실행 함수
# ============================================================================

def run_multiple_experiments(config_file="experiments.json"):
    """
    여러 실험을 순차적으로 실행하고 결과를 종합
    
    기능:
    - JSON 설정 파일에서 여러 실험 설정 로드
    - 각 실험을 순차적으로 실행
    - 전체 실험 결과 요약 생성
    - 성능 순위표 생성
    - 종합 결과를 JSON 파일로 저장
    
    Args:
        config_file (str): 실험 설정 JSON 파일 경로
        
    Returns:
        list: 각 실험의 결과 딕셔너리 리스트
    """
    # 실험 설정 로드
    experiments = load_experiments_config(config_file)
    
    if not experiments:
        print("❌ 실행할 실험이 없습니다.")
        return
    
    print(f"🔬 총 {len(experiments)}개의 실험을 시작합니다.")
    print(f"⏰ 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    total_start_time = time.time()
    
    # 각 실험 순차 실행
    for i, experiment in enumerate(experiments):
        try:
            result = run_single_experiment(experiment, i)
            results.append(result)
            
            if result['status'] == 'success':
                print(f"✅ {experiment['name']} 완료 - 정확도: {result['accuracy']:.2f}%")
                print(f"   📊 예측 데이터 수: {result['prediction_summary']['total_predictions']}개")
            else:
                print(f"❌ {experiment['name']} 실패: {result.get('message', '알 수 없는 오류')}")
                
        except Exception as e:
            print(f"❌ {experiment['name']} 실행 중 오류: {str(e)}")
            results.append({"status": "error", "message": str(e), "experiment_name": experiment['name']})
    
    total_end_time = time.time()
    total_time = round(total_end_time - total_start_time, 2)
    
    # ========================================================================
    # 전체 실험 결과 요약 출력
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"📊 실험 결과 요약")
    print(f"{'='*60}")
    print(f"⏱️  총 실행 시간: {total_time}초")
    print(f"✅ 성공: {len([r for r in results if r['status'] == 'success'])}개")
    print(f"❌ 실패: {len([r for r in results if r['status'] == 'error'])}개")
    
    # 성공한 실험들만 필터링 및 정확도 순 정렬
    successful_results = [r for r in results if r['status'] == 'success']
    if successful_results:
        successful_results.sort(key=lambda x: x['accuracy'], reverse=True)
        print(f"\n🏆 정확도 순위:")
        for i, result in enumerate(successful_results, 1):
            print(f"{i}. {result['experiment_name']}: {result['accuracy']:.2f}% (MAPE: {result['mape']:.2f}%)")
            print(f"   📈 R² Score: {result.get('r2_score', 'N/A')}, 방향성 정확도: {result.get('direction_accuracy', 'N/A'):.1f}%")
    
    # ========================================================================
    # 종합 결과 JSON 구성
    # ========================================================================
    comprehensive_results = {
        # 실험 전체 요약
        "experiment_summary": {
            "total_experiments": len(experiments),
            "successful_experiments": len(successful_results),
            "failed_experiments": len(results) - len(successful_results),
            "total_execution_time_seconds": total_time,
            "start_timestamp": datetime.now().isoformat(),
            "completion_timestamp": datetime.now().isoformat()
        },
        # 성능 순위
        "performance_ranking": [
            {
                "rank": i + 1,
                "experiment_name": result['experiment_name'],
                "model_name": result['modelName'],
                "accuracy": result['accuracy'],
                "mape": result['mape'],
                "mae": result.get('mae', None),
                "rmse": result.get('rmse', None),
                "r2_score": result.get('r2_score', None),
                "direction_accuracy": result.get('direction_accuracy', None),
                "total_predictions": result['prediction_summary']['total_predictions'] if 'prediction_summary' in result else 0
            }
            for i, result in enumerate(successful_results)
        ],
        # 상세 결과
        "detailed_results": results,
        # 예측 파일 목록
        "prediction_files": [
            {
                "experiment_name": result['experiment_name'],
                "model_name": result['modelName'],
                "prediction_file_path": result.get('prediction_file', 'N/A'),
                "prediction_count": result['prediction_summary']['total_predictions'] if 'prediction_summary' in result else 0,
                "recent_predictions": result.get('recent_predictions', [])
            }
            for result in successful_results
        ]
    }
    
    # ========================================================================
    # 종합 결과를 JSON 파일로 저장
    # ========================================================================
    comprehensive_results_file = "comprehensive_experiment_results.json"
    with open(comprehensive_results_file, "w", encoding="utf-8") as f:
        json.dump(comprehensive_results, f, indent=2, ensure_ascii=False, default=convert_numpy_to_json_serializable)
    
    print(f"\n💾 종합 결과가 '{comprehensive_results_file}'에 저장되었습니다.")
    print(f"📁 개별 예측 결과는 'predictions/' 폴더에 저장되었습니다.")
    
    # 생성된 예측 파일 목록 출력
    if successful_results:
        print(f"\n📄 생성된 예측 파일 목록:")
        for result in successful_results:
            if 'prediction_file' in result:
                print(f"   - {result['prediction_file']}")
    
    return results

# ============================================================================
# 메인 실행부
# ============================================================================

if __name__ == "__main__":
    """
    프로그램 진입점
    
    사용법:
    1. experiments.json 파일에 실험 설정 작성
    2. 이 스크립트 실행
    3. 설정 파일명 입력 (기본값: experiments.json)
    4. 자동으로 모든 실험 수행 및 결과 저장
    """
    print("\n📖 멀티 실험 모드 설명:")
    print("   - experiments.json 파일의 설정에 따라 여러 실험을 순차 실행")
    print("   - 각 실험별로 모델 학습, 예측, 성능 평가를 자동화")
    print("   - 결과를 종합하여 성능 순위표 자동 생성")
    
    # 사용자로부터 설정 파일명 입력 받기
    config_file = input("설정 파일명 (기본값: experiments.json): ").strip() or "experiments.json"
    
    # 멀티 실험 실행
    results = run_multiple_experiments(config_file)
    
    # 최종 결과 안내
    if results and any(r['status'] == 'success' for r in results):
        print(f"\n🎉 모든 실험이 완료되었습니다!")
        print(f"📁 다음 파일들이 생성되었습니다:")
        print(f"   - comprehensive_experiment_results.json (종합 결과)")
        print(f"   - predictions/ 폴더 (개별 예측 파일들)")
        print(f"   - graphImage/ 폴더 (시각화 그래프들)")
        print(f"   - saved_models/ 폴더 (학습된 모델들)")