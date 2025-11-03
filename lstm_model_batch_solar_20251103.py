# -*- coding: utf-8 -*-
"""
Title   : 외부데이터 기반 LSTM 예측 및 멀티 실험 자동화 모듈 (예측값 JSON 기록 기능 추가)
Author  : 주성중 / (주)맵인어스
Description:
    - LSTM 모델을 사용한 전력 생산량 예측 학습 모듈
    - 특정 시간대(07:00~16:45) 데이터만 사용
    - 멀티 실험 자동화 지원 (JSON 설정 기반)
    - 예측 결과 및 성능 지표 자동 저장
    - PostgreSQL 데이터베이스 연동
    - 80/20 데이터 분할 적용
"""

# ============================================================================
# 환경 설정 및 라이브러리 임포트
# ============================================================================

import os
# TensorFlow 최적화 옵션 비활성화 (경고 메시지 억제)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# TensorFlow 로그 레벨 설정 (ERROR만 출력)
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
# 환경별 경로 설정
# ============================================================================

# Flask 환경 변수를 통해 로컬/서버 환경 구분
ENV = os.getenv('FLASK_ENV', 'local')
if ENV == 'local':
    root = "D:/work/lstm"
else:
    root = "/app/webfiles/lstm"

# 그래프 이미지 저장 경로 생성
graph_path = os.path.abspath(root + "/graphImage")
os.makedirs(graph_path, exist_ok=True)

# 학습된 모델 저장 경로 생성
model_path = os.path.abspath(root + "/saved_models")
os.makedirs(model_path, exist_ok=True)

# 예측 결과 JSON 저장 경로 생성
prediction_path = os.path.abspath(root + "/predictions")
os.makedirs(prediction_path, exist_ok=True)

# ============================================================================
# PostgreSQL 데이터베이스 연결
# ============================================================================

def get_db_engine():
    """
    SQLAlchemy 엔진 생성 함수
    
    Returns:
        SQLAlchemy Engine 객체 - PostgreSQL 데이터베이스 연결
    
    용도:
        - 데이터 로드 (학습용 시계열 데이터)
        - 실험 결과 저장 (모델 정보, 성능 지표)
    """
    connection_string = "postgresql://postgres:mapinus@10.10.10.201:5432/postgres"
    # connection_string = "postgresql://postgres:7926@localhost:5432/postgres"
    return create_engine(connection_string)

# ============================================================================
# 실험 설정 파일 로드
# ============================================================================

def load_experiments_config(config_file="experiments.json"):
    """
    실험 설정 JSON 파일 로드
    
    Args:
        config_file (str): 설정 파일 경로 (기본값: experiments.json)
    
    Returns:
        list: 실험 설정 리스트
            각 실험 설정은 다음 정보를 포함:
            - name: 실험명
            - modelName: 모델명
            - tablename: 데이터 테이블명
            - dateColumn: 날짜 컬럼명
            - studyColumns: 학습 컬럼들 (쉼표 구분)
            - targetColumn: 예측 대상 컬럼
            - r_epochs: 학습 에포크 수
            - r_batchSize: 배치 크기
            - r_validationSplit: 검증 데이터 비율
            - r_seqLen: 입력 시퀀스 길이
            - r_predDays: 예측할 미래 스텝 수
        []: 로드 실패 시 빈 리스트
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
# 데이터베이스에서 학습 데이터 로드 (특정 시간대 및 날짜 필터링)
# ============================================================================

def load_data_from_db(tablename, dateColumn, studyColumns):
    """
    데이터베이스에서 학습용 시계열 데이터 로드 (특정 날짜 제외)
    
    Args:
        tablename (str): 테이블명 (예: lstm_input_15m_new)
        dateColumn (str): 날짜 컬럼명 (예: time_point)
        studyColumns (str): 학습할 컬럼들 (쉼표로 구분)
    
    Returns:
        pandas.DataFrame: 시계열 순으로 정렬된 데이터
            - 시간대별 분포 정보 출력
            - NULL 값이 있는 행은 제외
        None: 로드 실패 시
    
    데이터 필터링:
        - dateColumn IS NOT NULL: 날짜가 있는 데이터만
        - 특정 날짜 제외 (이상치 날짜들):
          '06-02', '06-13~17', '06-20~21', '06-24~26', '06-28',
          '07-01', '07-08', '07-13~19', '07-21~22'
        - 시간대 분포 분석 및 출력
    
    이유:
        - 전력 생산량은 주로 07:00~16:45에 발생
        - 특정 날짜는 이상치 또는 특수한 상황 (공휴일, 설비 점검 등)
    """
    try:
        engine = get_db_engine()
        
        # SQL 쿼리 작성
        # - carbontwin 스키마의 지정된 테이블에서 데이터 조회
        # - 특정 날짜 제외 처리 (이상치 날짜들)
        # - 날짜 기준 오름차순 정렬
        query = f"""
        SELECT {studyColumns},{dateColumn}
        FROM carbontwin.{tablename}
        WHERE {dateColumn} IS NOT NULL
        AND TO_CHAR({dateColumn}, 'MM-DD') NOT IN (
            '06-02', '06-13', '06-14', '06-15', '06-16', '06-17',
            '06-20', '06-21', '06-24', '06-25', '06-26', '06-28',
            '07-01', '07-08', '07-13', '07-14', '07-15', '07-16',
            '07-17', '07-18', '07-19', '07-21', '07-22'
        )
        ORDER BY {dateColumn} ASC
        """
        
        # 데이터 로드
        data = pd.read_sql_query(query, engine)

        # ✅ 시간대 분포 확인 및 출력
        # 데이터의 시간 범위와 시간대별 데이터 수 분석
        if dateColumn in data.columns and len(data) > 0:
            data[dateColumn] = pd.to_datetime(data[dateColumn])
            hours = data[dateColumn].dt.hour
            print(f"   📊 시간 범위: {hours.min()}시 ~ {hours.max()}시")
            
            # 시간대별 데이터 개수 집계
            hour_counts = hours.value_counts().sort_index()
            print(f"   📊 시간대별 데이터 수:")
            for hour, count in hour_counts.items():
                print(f"      {hour:2d}시: {count:5d}개")
        
        return data
        
    except Exception as e:
        print(f"❌ 데이터베이스 오류: {str(e)}")
        return None

# ============================================================================
# JSON 직렬화 지원 함수
# ============================================================================

def convert_numpy_to_json_serializable(obj):
    """
    NumPy 배열과 특수 타입을 JSON 직렬화 가능한 형태로 변환
    
    Args:
        obj: 변환할 객체
            - np.ndarray: NumPy 배열
            - np.integer: NumPy 정수형
            - np.floating: NumPy 실수형
            - pd.Timestamp: Pandas 타임스탬프
            - datetime: Python datetime 객체
    
    Returns:
        JSON 직렬화 가능한 Python 기본 타입
            - list: NumPy 배열 → Python 리스트
            - int: NumPy 정수 → Python int
            - float: NumPy 실수 → Python float
            - str: Timestamp/datetime → ISO 형식 문자열
    
    용도:
        - 예측 결과를 JSON 파일로 저장할 때
        - 실험 결과를 JSON으로 직렬화할 때
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    else:
        return obj

# ============================================================================
# 예측 결과 JSON 저장 함수
# ============================================================================

def save_predictions_to_json(modelName, dates, actual_values, predicted_values, target_column):
    """
    모델의 예측 결과를 JSON 파일로 저장
    
    Args:
        modelName (str): 모델명 (파일명에 사용)
        dates: 예측 시점들 (pandas Series 또는 array)
        actual_values (array): 실제값 배열
        predicted_values (array): 예측값 배열
        target_column (str): 타겟 컬럼명
    
    Returns:
        dict: 예측 결과 요약 정보
            - model_name: 모델명
            - target_column: 타겟 컬럼
            - prediction_count: 예측 개수
            - timestamp: 저장 시각
            - statistics: 통계 정보 (최소, 최대, 평균, MAE, RMSE)
            - predictions: 각 시점별 예측 상세 정보
        None: 저장 실패 시
    
    저장 내용:
        각 예측 시점마다:
        - index: 순서
        - date: 날짜/시간
        - actual_value: 실제값
        - predicted_value: 예측값
        - difference: 차이 (예측 - 실제)
        - percentage_error: 백분율 오차
    
    저장 위치:
        {prediction_path}/{modelName}_predictions.json
    """
    try:
        # 예측 데이터 구성 - 각 시점별 실제값과 예측값을 비교
        predictions_data = []
        
        for i in range(len(actual_values)):
            prediction_record = {
                "index": i,
                "date": convert_numpy_to_json_serializable(dates.iloc[i] if hasattr(dates, 'iloc') else dates[i]),
                "actual_value": convert_numpy_to_json_serializable(actual_values[i]),
                "predicted_value": convert_numpy_to_json_serializable(predicted_values[i]),
                "difference": convert_numpy_to_json_serializable(predicted_values[i] - actual_values[i]),
                "percentage_error": convert_numpy_to_json_serializable(
                    abs((predicted_values[i] - actual_values[i]) / actual_values[i] * 100) if actual_values[i] != 0 else 0
                )
            }
            predictions_data.append(prediction_record)
        
        # 저장 파일 경로 생성
        prediction_file_path = os.path.join(prediction_path, f"{modelName}_predictions.json")
        
        # 예측 결과 요약 정보 구성
        prediction_summary = {
            "model_name": modelName,
            "target_column": target_column,
            "prediction_count": len(predictions_data),
            "timestamp": datetime.now().isoformat(),
            "statistics": {
                "actual_min": convert_numpy_to_json_serializable(np.min(actual_values)),
                "actual_max": convert_numpy_to_json_serializable(np.max(actual_values)),
                "actual_mean": convert_numpy_to_json_serializable(np.mean(actual_values)),
                "predicted_min": convert_numpy_to_json_serializable(np.min(predicted_values)),
                "predicted_max": convert_numpy_to_json_serializable(np.max(predicted_values)),
                "predicted_mean": convert_numpy_to_json_serializable(np.mean(predicted_values)),
                "mean_absolute_error": convert_numpy_to_json_serializable(np.mean(np.abs(predicted_values - actual_values))),
                "rmse": convert_numpy_to_json_serializable(np.sqrt(np.mean((predicted_values - actual_values) ** 2)))
            },
            "predictions": predictions_data
        }
        
        # JSON 파일로 저장
        with open(prediction_file_path, 'w', encoding='utf-8') as f:
            json.dump(prediction_summary, f, indent=2, ensure_ascii=False)
        
        print(f"💾 예측 결과가 저장되었습니다: {prediction_file_path}")
        return prediction_summary
        
    except Exception as e:
        print(f"❌ 예측 결과 저장 중 오류: {str(e)}")
        return None

# ============================================================================
# 실험 결과 데이터베이스 저장 함수
# ============================================================================

def save_experiment_to_db(result, config, is_new_model):
    """
    실험 결과를 PostgreSQL 데이터베이스에 저장
    
    Args:
        result (dict): 실험 결과 딕셔너리
            - modelName: 모델명
            - accuracy, mape, rmse, r2_score: 성능 지표
            - training_loss_img, total_graph_img, diff_graph_img: 그래프 경로
            - prediction_file: 예측 파일 경로
            - execution_time: 실행 시간
        config (dict): 실험 설정 정보
        is_new_model (bool): 신규 모델 여부
    
    Returns:
        bool: 저장 성공 여부
    
    저장 테이블:
        1. lstm_model 테이블:
           - 신규 모델인 경우 모델 기본 정보 등록
           - model_name, target_column, sequence_length 등
        
        2. lstm_experiment 테이블:
           - 실험 결과 저장
           - model_id (외래키), 성능 지표, 그래프 경로 등
    
    프로세스:
        1. 모델 존재 여부 확인
        2. 신규 모델인 경우 lstm_model에 등록
        3. model_id 조회
        4. 실험 결과를 lstm_experiment에 저장
    """
    try:
        engine = get_db_engine()
        model_name = result.get('modelName')
        
        # 신규 모델인 경우 lstm_model 테이블에 등록
        if is_new_model:
            # 기존 모델 존재 여부 확인
            check_query = f"SELECT model_id FROM carbontwin.lstm_model WHERE model_name = '{model_name}'"
            existing = pd.read_sql_query(check_query, engine)
            
            if existing.empty:
                # 모델 기본 정보 구성
                model_data = {
                    'model_name': model_name,
                    'target_column': config.get('targetColumn'),
                    'date_column': config.get('dateColumn'),
                    'study_columns': config.get('studyColumns'),
                    'epochs': config.get('r_epochs'),
                    'batch_size': config.get('r_batchSize'),
                    'validation_split': config.get('r_validationSplit'),
                    'sequence_length': config.get('r_seqLen'),
                    'prediction_days': config.get('r_predDays'),
                    'created_at': datetime.now()
                }
                
                # lstm_model 테이블에 삽입
                df_model = pd.DataFrame([model_data])
                df_model.to_sql('lstm_model', engine, schema='carbontwin',
                              if_exists='append', index=False)
                print(f"✅ 신규 모델 등록: {model_name}")
            else:
                print(f"ℹ️  기존 모델 사용: {model_name}")
        
        # model_id 조회
        query = f"SELECT model_id FROM carbontwin.lstm_model WHERE model_name = '{model_name}'"
        model_id = pd.read_sql_query(query, engine).iloc[0]['model_id']
        
        # 실험 결과 데이터 구성
        experiment_data = {
            'model_id': model_id,
            'experiment_name': result.get('experiment_name', config.get('name')),
            'accuracy': result.get('accuracy'),
            'mape': result.get('mape'),
            'rmse': result.get('rmse'),
            'r2_score': result.get('r2_score'),
            'model_file_path': os.path.abspath(os.path.join(model_path, f"{model_name}.h5")),
            'training_loss_img_path': os.path.abspath(os.path.join(root, result.get('training_loss_img'))),
            'total_graph_img_path': os.path.abspath(os.path.join(root, result.get('total_graph_img'))),
            'diff_graph_img_path': os.path.abspath(os.path.join(root, result.get('diff_graph_img'))),
            'prediction_file_path': os.path.abspath(os.path.join(root, result.get('prediction_file'))),
            'execution_time_seconds': result.get('execution_time'),
            'status': result.get('status'),
            'config_json': json.dumps(config, ensure_ascii=False),
            'created_at': datetime.now()
        }
        
        # lstm_experiment 테이블에 삽입
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
    단일 실험을 실행하고 결과를 데이터베이스에 저장
    
    Args:
        experiment_config (dict): 실험 설정 정보
        experiment_index (int): 실험 순서 (출력용)
    
    Returns:
        dict: 실험 결과
            - status: 성공/실패 상태
            - execution_time: 실행 시간
            - 성능 지표들 (accuracy, mape, rmse 등)
    
    실행 순서:
        1. 실험 시작 알림
        2. 데이터베이스에서 데이터 로드
        3. LSTM 모델 학습 실행 (lstmFinance 함수 호출)
        4. 실행 시간 기록
        5. 성공 시 데이터베이스에 결과 저장
        6. 결과 반환
    """
    print(f"\n{'='*60}")
    print(f"🚀 실험 {experiment_index + 1} 시작: {experiment_config['name']}")
    print(f"{'='*60}")
    
    # 1. 데이터 로드
    data = load_data_from_db(
        experiment_config['tablename'],
        experiment_config['dateColumn'], 
        experiment_config['studyColumns']
    )
    
    if data is None:
        return {"status": "error", "message": "데이터 로드 실패"}
    
    # 2. 학습 실행 및 시간 측정
    start_time = time.time()
    result = lstmFinance(data, experiment_config)
    end_time = time.time()
    
    # 3. 실행 시간 및 실험명 추가
    result['execution_time'] = round(end_time - start_time, 2)
    result['experiment_name'] = experiment_config['name']
    
    print(f"⏱️  실험 완료 시간: {result['execution_time']}초")
    
    # 4. 성공 시 데이터베이스에 저장
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
# LSTM 모델 학습 메인 함수
# ============================================================================

def lstmFinance(lstmData, config):
    """
    LSTM 모델 학습 함수 (전력 생산량 예측용)
    
    특징:
    - 80/20 데이터 분할 적용
    - 특정 시간대 데이터만 사용 (07:00~16:45)
    - StandardScaler를 사용한 데이터 정규화
    - 예측 결과를 JSON 파일로 저장
    - 전체 예측 결과 콘솔 출력
    
    Args:
        lstmData (DataFrame): 학습 데이터
        config (dict): 모델 학습 설정
            - modelName: 모델명
            - dateColumn: 날짜 컬럼명
            - studyColumns: 학습 컬럼들
            - targetColumn: 예측 타겟
            - r_epochs: 에포크 수
            - r_batchSize: 배치 크기
            - r_validationSplit: 검증 데이터 비율
            - r_seqLen: 시퀀스 길이
            - r_predDays: 예측 스텝 수
    
    Returns:
        dict: 학습 결과
            - status: 성공/실패
            - modelName: 모델명
            - 성능 지표 (accuracy, mape, rmse, r2_score 등)
            - 그래프 파일 경로들
            - 예측 결과 파일 경로
    
    학습 프로세스:
        1. 설정 파라미터 추출
        2. 데이터 검증
        3. 날짜 컬럼 처리 및 시간대 확인
        4. 데이터 스케일링
        5. 학습/테스트 분할 (80/20)
        6. 시퀀스 데이터 생성
        7. 모델 생성 또는 로드
        8. 학습 (새 모델인 경우)
        9. 예측 수행
        10. 역정규화
        11. 성능 지표 계산
        12. 전체 예측 결과 출력
        13. 시각화 그래프 생성
        14. 결과 저장 및 반환
    """
    
    # TensorFlow eager execution 활성화 (디버깅 용이)
    if not tf.executing_eagerly():
        tf.config.run_functions_eagerly(True)

    # ====================================================================
    # 1단계: 설정 파라미터 추출
    # ====================================================================
    modelName = config['modelName']
    dateColumn = config['dateColumn']
    studyColumns = config['studyColumns']
    targetColumn = config['targetColumn']
    r_epochs = config['r_epochs']
    r_batchSize = config['r_batchSize']
    r_validationSplit = config['r_validationSplit']
    r_seqLen = config['r_seqLen']
    r_predDays = config['r_predDays']

    # 결과 파일 경로 설정
    training_loss_path = os.path.join(graph_path, f"{modelName}_trainingLoss.png")
    total_graph_path = os.path.join(graph_path, f"{modelName}_totalgraph.png")
    diff_graph_path = os.path.join(graph_path, f"{modelName}_diffgraph.png")
    model_file_path = os.path.join(model_path, f"{modelName}.h5")

    stock_data = lstmData
    
    # ====================================================================
    # 2단계: 데이터 검증
    # ====================================================================
    if stock_data.empty:
        return {"status": "error", "message": "데이터가 비어있습니다."}
    
    print(f"\n📊 로드된 데이터 정보:")
    print(f"   - 총 데이터 수: {len(stock_data)}개")
    
    # 학습 컬럼 리스트 생성 및 타겟 컬럼 검증
    study_columns_list = [col.strip() for col in studyColumns.split(',')]
    if targetColumn not in study_columns_list:
        return {"status": "error", "message": f"타겟 컬럼 '{targetColumn}'이 학습 컬럼에 없습니다."}

    # ====================================================================
    # 3단계: 날짜 컬럼 처리 및 시간대 확인
    # ====================================================================
    # 날짜 컬럼이 있으면 사용, 없으면 가상 날짜 생성
    if dateColumn in stock_data.columns:
        dates = pd.to_datetime(stock_data[dateColumn], errors='coerce')
        
        # ✅ 시간 범위 확인 및 출력
        # 데이터가 07:00~16:45 범위에 있는지 확인
        hours = dates.dt.hour
        print(f"   - 시간 범위: {hours.min()}시 ~ {hours.max()}시")
        print(f"   - 고유 시간대: {sorted(hours.unique())}")
    else:
        # 15분 간격 가상 날짜 생성
        dates = pd.date_range(start='2023-01-01', periods=len(stock_data), freq='15T')
        print(f"⚠️ 날짜 컬럼 '{dateColumn}'이 없어서 가상 날짜를 생성했습니다.")
    
    # ====================================================================
    # 4단계: 데이터 준비 및 스케일링
    # ====================================================================
    # 원본 타겟 값 저장 (나중에 그래프에 사용)
    original_open = stock_data[targetColumn].values
    # 학습용 데이터 준비 (모든 study columns)
    stock_data_for_training = stock_data[study_columns_list].astype(float)

    # 데이터 표준화 (평균 0, 분산 1로 변환)
    # 이유: LSTM이 정규화된 데이터에서 더 잘 학습함
    scaler = StandardScaler()
    stock_data_scaled = scaler.fit_transform(stock_data_for_training)

    # ====================================================================
    # 5단계: 80/20 split (학습 80%, 테스트 20%)
    # ====================================================================
    # 이전 90/10에서 변경하여 테스트 데이터 비중 증가
    split_index = int(len(stock_data_scaled) * 0.8)
    train_data_scaled = stock_data_scaled[:split_index]
    test_data_scaled = stock_data_scaled[split_index:]
    test_dates = dates[split_index:]

    # 학습 파라미터 설정
    pred_days = int(r_predDays)  # 예측할 미래 스텝 수
    seq_len = int(r_seqLen)  # 입력 시퀀스 길이
    input_dim = stock_data_for_training.shape[1]  # 입력 피처 개수
    target_idx = study_columns_list.index(targetColumn)  # 타겟 컬럼의 인덱스

    # ====================================================================
    # 6단계: 데이터 충분성 검증
    # ====================================================================
    # 시퀀스를 생성하기에 충분한 데이터가 있는지 확인
    print(f"\n🔍 시퀀스 생성 검증:")
    print(f"   - 전체 데이터: {len(stock_data_scaled)}개")
    print(f"   - 학습 데이터: {len(train_data_scaled)}개")
    print(f"   - 테스트 데이터: {len(test_data_scaled)}개")
    print(f"   - 시퀀스 길이(seq_len): {seq_len}")
    print(f"   - 예측 일수(pred_days): {pred_days}")
    
    min_required = seq_len + pred_days
    print(f"   - 필요한 최소 데이터: {min_required}개")
    
    # 학습 데이터 충분성 검사
    if len(train_data_scaled) < min_required:
        error_msg = f"학습 데이터 부족: {len(train_data_scaled)}개 (최소 {min_required}개 필요)"
        print(f"❌ {error_msg}")
        return {"status": "error", "message": error_msg}
    
    # 테스트 데이터 충분성 검사
    if len(test_data_scaled) < min_required:
        error_msg = f"테스트 데이터 부족: {len(test_data_scaled)}개 (최소 {min_required}개 필요)"
        print(f"❌ {error_msg}")
        return {"status": "error", "message": error_msg}

    # ====================================================================
    # 7단계: 시퀀스 데이터 생성
    # ====================================================================
    # LSTM 입력 형태로 변환: (샘플 수, 시퀀스 길이, 피처 수)
    trainX, trainY, testX, testY = [], [], [], []
    
    # 생성 가능한 시퀀스 범위 계산
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
    
    # 학습 시퀀스 생성
    for i in train_range:
        # X: 과거 seq_len 스텝의 모든 피처
        trainX.append(train_data_scaled[i - seq_len:i, 0:input_dim])
        # Y: pred_days 후의 타겟 값
        trainY.append(train_data_scaled[i + pred_days - 1:i + pred_days, target_idx])

    # 테스트 시퀀스 생성
    for i in test_range:
        testX.append(test_data_scaled[i - seq_len:i, 0:input_dim])
        testY.append(test_data_scaled[i + pred_days - 1:i + pred_days, target_idx])

    # NumPy 배열로 변환
    trainX, trainY = np.array(trainX), np.array(trainY)
    testX, testY = np.array(testX), np.array(testY)

    print(f"✅ 시퀀스 생성 완료:")
    print(f"   - trainX: {trainX.shape}, trainY: {trainY.shape}")
    print(f"   - testX: {testX.shape}, testY: {testY.shape}")

    # ====================================================================
    # 8단계: 모델 생성 또는 로드
    # ====================================================================
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

        # LSTM 모델 아키텍처 구성
        model = Sequential([
            Input(shape=(trainX.shape[1], trainX.shape[2])),  # (시퀀스 길이, 피처 수)
            LSTM(64, return_sequences=True),  # 첫 번째 LSTM 레이어 (64 유닛)
            LSTM(32, return_sequences=False),  # 두 번째 LSTM 레이어 (32 유닛)
            Dense(trainY.shape[1])  # 출력 레이어
        ])

        # 모델 컴파일
        # optimizer: Adam (학습률 0.01)
        # loss: MSE (평균제곱오차)
        model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')

        # ================================================================
        # 학습 진행 모니터링 콜백 클래스
        # ================================================================
        class TrainingCallback(Callback):
            """
            학습 과정을 모니터링하는 콜백
            
            기능:
            - 에포크별 손실 출력
            - 진행률 표시 (프로그레스 바)
            - 검증 손실 개선/악화 알림
            """
            def __init__(self, total_epochs, batch_size):
                super().__init__()
                self.total_epochs = total_epochs
                self.batch_size = batch_size
                self.prev_val_loss = None
                
            def on_train_begin(self, logs=None):
                """학습 시작 시 호출"""
                print(f"🚀 모델 학습 시작 - 총 {self.total_epochs} 에포크")
                print(f"📊 배치 크기: {self.batch_size}")
                
            def on_epoch_begin(self, epoch, logs=None):
                """각 에포크 시작 시 호출"""
                print(f"\n⏳ Epoch {epoch + 1}/{self.total_epochs} 시작...")
                
            def on_epoch_end(self, epoch, logs=None):
                """각 에포크 종료 시 호출"""
                logs = logs or {}
                loss = logs.get('loss', 0)
                val_loss = logs.get('val_loss', 0)
                
                # 진행률 계산 및 프로그레스 바 표시
                progress = (epoch + 1) / self.total_epochs * 100
                bar_length = 30
                filled_length = int(bar_length * (epoch + 1) // self.total_epochs)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                
                print(f"✅ Epoch {epoch + 1}/{self.total_epochs} [{bar}] {progress:.1f}%")
                print(f"   📉 Loss: {loss:.6f} | Val_Loss: {val_loss:.6f}")
                
                # 검증 손실 변화 분석
                if epoch > 0 and self.prev_val_loss is not None:
                    if val_loss < self.prev_val_loss:
                        print(f"   📈 검증 손실 개선: {self.prev_val_loss:.6f} → {val_loss:.6f}")
                    elif val_loss > self.prev_val_loss * 1.1:  # 10% 이상 증가
                        print(f"   ⚠️  검증 손실 증가: {self.prev_val_loss:.6f} → {val_loss:.6f}")
                
                self.prev_val_loss = val_loss
                
            def on_train_end(self, logs=None):
                """학습 완료 시 호출"""
                print(f"\n🎉 학습 완료!")

        # 모델 학습 실행
        history = model.fit(
            trainX, trainY,
            epochs=int(r_epochs),
            batch_size=int(r_batchSize),
            validation_split=float(r_validationSplit),
            verbose=1,
            callbacks=[TrainingCallback(int(r_epochs), int(r_batchSize))]
        )

        # 학습된 모델 저장
        model.save(model_file_path)
        print("✅ 모델 저장 완료")

        # 학습 손실 그래프 생성 및 저장
        plt.figure(figsize=(12, 4))
        plt.plot(history.history['loss'], label='Training loss')
        plt.plot(history.history['val_loss'], label='Validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{modelName} - Training Loss')
        plt.legend()
        plt.savefig(training_loss_path)
        plt.close()

    # ====================================================================
    # 9단계: 예측 수행
    # ====================================================================
    print(f"\n🔮 예측 수행 중...")
    print(f"📊 예측할 샘플 수: {len(testX)}")
    
    # 배치 단위로 예측 (메모리 효율성)
    batch_size_pred = 32
    predictions = []
    total_batches = (len(testX) + batch_size_pred - 1) // batch_size_pred
    
    for i in range(0, len(testX), batch_size_pred):
        batch_end = min(i + batch_size_pred, len(testX))
        batch_data = testX[i:batch_end]
        
        # 배치 예측
        batch_pred = model.predict(batch_data, verbose=0)
        predictions.append(batch_pred)
        
        # 진행률 표시
        current_batch = (i // batch_size_pred) + 1
        progress = current_batch / total_batches * 100
        bar_length = 25
        filled_length = int(bar_length * current_batch // total_batches)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        print(f"\r⏳ 예측 진행: [{bar}] {progress:.1f}% ({current_batch}/{total_batches} 배치)", end='', flush=True)
    
    # 모든 배치 예측을 하나로 합침
    prediction = np.vstack(predictions)
    print(f"\n✅ 예측 완료! 총 {len(prediction)}개 샘플 예측됨")

    # ====================================================================
    # 10단계: 예측 결과 역변환 (역정규화)
    # ====================================================================
    # 정규화된 값을 원래 스케일로 되돌림
    
    # 예측값 역변환
    mean_values_pred = np.repeat(scaler.mean_[np.newaxis, :], prediction.shape[0], axis=0)
    mean_values_pred[:, target_idx] = np.squeeze(prediction)
    y_pred = scaler.inverse_transform(mean_values_pred)[:, target_idx]

    # 실제값 역변환
    mean_values_testY = np.repeat(scaler.mean_[np.newaxis, :], testY.shape[0], axis=0)
    mean_values_testY[:, target_idx] = np.squeeze(testY)
    testY_original = scaler.inverse_transform(mean_values_testY)[:, target_idx]
    
    # 유효한 테스트 날짜 추출
    valid_test_dates = test_dates[seq_len : seq_len + len(testY_original)]

    # ====================================================================
    # 11단계: 예측 결과 JSON 저장
    # ====================================================================
    print(f"\n💾 예측 결과를 JSON 파일로 저장 중...")
    prediction_summary = save_predictions_to_json(
        modelName, 
        valid_test_dates, 
        testY_original, 
        y_pred, 
        targetColumn
    )

    # ====================================================================
    # 12단계: 시각화 그래프 생성
    # ====================================================================
    
    # 전체 기간 그래프 (원본 + 실제 + 예측)
    plt.figure(figsize=(15, 5))
    plt.plot(dates, original_open, color='green', label=f'Original {targetColumn}', alpha=0.7)
    plt.plot(valid_test_dates, testY_original, color='blue', label=f'Actual {targetColumn}')
    plt.plot(valid_test_dates, y_pred, color='red', linestyle='--', label=f'Predicted {targetColumn}')
    plt.xlabel(dateColumn)
    plt.ylabel(f'{targetColumn} Value')
    plt.title(f'{modelName} - Prediction Results')
    plt.legend()
    plt.savefig(total_graph_path)
    plt.close()

    # 최근 50개 포인트 확대 그래프
    zoom_start = max(0, len(valid_test_dates) - 50)
    plt.figure(figsize=(15, 5))
    plt.plot(valid_test_dates[zoom_start:], testY_original[zoom_start:], color='blue', label=f'Actual {targetColumn}')
    plt.plot(valid_test_dates[zoom_start:], y_pred[zoom_start:], color='red', linestyle='--', label=f'Predicted {targetColumn}')
    plt.xlabel(dateColumn)
    plt.ylabel(f'{targetColumn} Value')
    plt.title(f'{modelName} - Recent Predictions (Last 50 points)')
    plt.legend()
    plt.savefig(diff_graph_path)
    plt.close()

    # ====================================================================
    # 13단계: 성능 지표 계산
    # ====================================================================
    print(f"\n📈 성능 평가 중...")
    
    # ====================================================================
    # MAPE 계산 함수 (상세 출력 포함)
    # ====================================================================
    def mean_absolute_percentage_error(y_true, y_pred, valid_test_dates):
        """
        MAPE (Mean Absolute Percentage Error) 계산 함수
        
        Args:
            y_true: 실제값
            y_pred: 예측값
            valid_test_dates: 날짜 정보
        
        Returns:
            float: MAPE 값 (%)
                   임계값 초과 데이터가 없으면 999.0 반환
        
        특징:
            - 임계값(eps=9) 이상의 데이터만 사용하여 MAPE 계산
            - 모든 예측 결과를 표 형태로 출력
            - 오차 분석 (평균, 표준편차, 최대/최소 오차)
        """
        eps = 9  # 임계값 (이 값보다 작은 실제값은 MAPE 계산에서 제외)
        mask = y_true > eps
        
        print(f"\n📊 MAPE 계산 정보:")
        print(f"   - 임계값(eps): {eps}")
        print(f"   - 전체 데이터: {len(y_true)}개")
        print(f"   - 임계값 초과 데이터: {np.sum(mask)}개")
        
        if np.sum(mask) == 0:
            print("   ⚠️ 임계값을 초과하는 데이터가 없습니다.")
            return 999.0
        
        # ✅ 임계값 초과하는 모든 데이터의 예측값 출력
        filtered_dates = valid_test_dates[mask]
        filtered_true = y_true[mask]
        filtered_pred = y_pred[mask]
        
        print(f"\n📋 임계값 초과 데이터 전체 ({len(filtered_true)}개):")
        print(f"{'='*90}")
        print(f"{'날짜/시간':<25} {'실제값':>12} {'예측값':>12} {'오차':>12} {'오차율(%)':>12}")
        print(f"{'-'*90}")
        
        # 각 데이터 포인트별 상세 정보 출력
        for i in range(len(filtered_true)):
            date_str = filtered_dates.iloc[i].strftime('%Y-%m-%d %H:%M:%S') if hasattr(filtered_dates.iloc[i], 'strftime') else str(filtered_dates.iloc[i])
            true_val = filtered_true[i]
            pred_val = filtered_pred[i]
            error = pred_val - true_val
            error_pct = abs(error / true_val * 100)
            
            print(f"{date_str:<25} {true_val:>12.4f} {pred_val:>12.4f} {error:>12.4f} {error_pct:>12.2f}")
        
        print(f"{'='*90}")
        
        # MAPE 계산
        mape_value = np.mean(np.abs((filtered_pred - filtered_true) / filtered_true)) * 100
        print(f"\n   ✅ 계산된 MAPE: {mape_value:.2f}%")
        
        # 추가 오차 분석
        errors = filtered_pred - filtered_true
        print(f"\n📊 오차 분석:")
        print(f"   - 평균 오차: {np.mean(errors):.4f}")
        print(f"   - 오차 표준편차: {np.std(errors):.4f}")
        print(f"   - 최대 과대예측: {np.max(errors):.4f}")
        print(f"   - 최대 과소예측: {np.min(errors):.4f}")
        print(f"   - 과대예측 비율: {np.sum(errors > 0) / len(errors) * 100:.1f}%")
        print(f"   - 과소예측 비율: {np.sum(errors < 0) / len(errors) * 100:.1f}%")
        
        return mape_value

    # sklearn 사용 가능 여부 확인
    try:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        sklearn_available = True
    except ImportError:
        print("⚠️ scikit-learn이 설치되지 않았습니다. 기본 지표만 계산합니다.")
        sklearn_available = False
    
    # 주요 성능 지표 계산
    mape = mean_absolute_percentage_error(testY_original, y_pred, valid_test_dates)
    accuracy = 100 - mape if not np.isnan(mape) else 0
    
    # 추가 지표들 계산
    if sklearn_available:
        # sklearn 사용
        mse = mean_squared_error(testY_original, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(testY_original, y_pred)
        r2 = r2_score(testY_original, y_pred)
    else:
        # 수동 계산
        mse = np.mean((testY_original - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(testY_original - y_pred))
        
        # R² 수동 계산
        ss_res = np.sum((testY_original - y_pred) ** 2)  # 잔차 제곱합
        ss_tot = np.sum((testY_original - np.mean(testY_original)) ** 2)  # 총 제곱합
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # 방향성 정확도 (상승/하락 방향 예측 정확도)
    if len(testY_original) > 1:
        actual_direction = np.diff(testY_original) > 0  # 실제 증가/감소
        pred_direction = np.diff(y_pred) > 0  # 예측 증가/감소
        direction_accuracy = np.mean(actual_direction == pred_direction) * 100
    else:
        direction_accuracy = 0
    
    # ====================================================================
    # 14단계: 전체 예측 결과 출력
    # ====================================================================
    # ✅ 전체 데이터의 예측 결과 모두 출력
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
    # 15단계: 성능 지표 요약 출력
    # ====================================================================
    print(f"\n📊 모델 성능 결과:")
    print(f"   🎯 MAPE: {mape:.2f}%")
    print(f"   📈 정확도: {accuracy:.2f}%")
    print(f"   📏 MAE: {mae:.4f}")
    print(f"   📐 RMSE: {rmse:.4f}")
    print(f"   🔍 R² Score: {r2:.4f}")
    print(f"   🧭 방향성 정확도: {direction_accuracy:.2f}%")
    
    # 성능 등급 평가
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
    
    # 예측 범위 분석
    pred_min, pred_max = np.min(y_pred), np.max(y_pred)
    actual_min, actual_max = np.min(testY_original), np.max(testY_original)
    print(f"\n📊 예측값 범위 분석:")
    print(f"   실제값 범위: {actual_min:.3f} ~ {actual_max:.3f}")
    print(f"   예측값 범위: {pred_min:.3f} ~ {pred_max:.3f}")
    
    # 과예측/소예측 비율
    over_predict = np.sum(y_pred > testY_original) / len(y_pred) * 100
    under_predict = 100 - over_predict
    print(f"   과예측 비율: {over_predict:.1f}%")
    print(f"   소예측 비율: {under_predict:.1f}%")

    # (중복된 성능 지표 계산 코드 - 제거하지 않고 유지)
    try:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        sklearn_available = True
    except ImportError:
        print("⚠️ scikit-learn이 설치되지 않았습니다. 기본 지표만 계산합니다.")
        sklearn_available = False
    
    mape = mean_absolute_percentage_error(testY_original, y_pred, valid_test_dates)
    accuracy = 100 - mape if not np.isnan(mape) else 0
    
    if sklearn_available:
        mse = mean_squared_error(testY_original, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(testY_original, y_pred)
        r2 = r2_score(testY_original, y_pred)
    else:
        mse = np.mean((testY_original - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(testY_original - y_pred))
        
        ss_res = np.sum((testY_original - y_pred) ** 2)
        ss_tot = np.sum((testY_original - np.mean(testY_original)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    if len(testY_original) > 1:
        actual_direction = np.diff(testY_original) > 0
        pred_direction = np.diff(y_pred) > 0
        direction_accuracy = np.mean(actual_direction == pred_direction) * 100
    else:
        direction_accuracy = 0
    
    # (중복 출력 - 제거하지 않고 유지)
    print(f"\n📊 모델 성능 결과:")
    print(f"   🎯 MAPE: {mape:.2f}%")
    print(f"   📈 정확도: {accuracy:.2f}%")
    print(f"   📏 MAE: {mae:.4f}")
    print(f"   📐 RMSE: {rmse:.4f}")
    print(f"   🔍 R² Score: {r2:.4f}")
    print(f"   🧭 방향성 정확도: {direction_accuracy:.2f}%")
    
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
    
    pred_min, pred_max = np.min(y_pred), np.max(y_pred)
    actual_min, actual_max = np.min(testY_original), np.max(testY_original)
    print(f"\n📊 예측값 범위 분석:")
    print(f"   실제값 범위: {actual_min:.3f} ~ {actual_max:.3f}")
    print(f"   예측값 범위: {pred_min:.3f} ~ {pred_max:.3f}")
    
    over_predict = np.sum(y_pred > testY_original) / len(y_pred) * 100
    under_predict = 100 - over_predict
    print(f"   과예측 비율: {over_predict:.1f}%")
    print(f"   소예측 비율: {under_predict:.1f}%")

    # ====================================================================
    # 16단계: 설정 및 스케일러 저장
    # ====================================================================
    # 모델 설정을 JSON으로 저장 (나중에 예측 시 사용)
    with open(os.path.join(model_path, f"{modelName}_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    # 스케일러 저장 (예측 시 동일한 정규화 적용 필요)
    joblib.dump(scaler, os.path.join(model_path, f"{modelName}_scaler.pkl"))

    # ====================================================================
    # 17단계: 결과 반환
    # ====================================================================
    # 예측 데이터를 포함한 반환값 구성
    result = {
        "status": "success",
        "modelName": modelName,
        "training_loss_img": f"graphImage/{modelName}_trainingLoss.png",
        "total_graph_img": f"graphImage/{modelName}_totalgraph.png",
        "diff_graph_img": f"graphImage/{modelName}_diffgraph.png",
        "mape": round(mape, 2),
        "accuracy": round(accuracy, 2),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2_score": round(r2, 4),
        "direction_accuracy": round(direction_accuracy, 2),
        "prediction_file": f"predictions/{modelName}_predictions.json",
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
    
    # 최근 N개 예측값을 직접 결과에 포함 (최근 10개)
    recent_predictions_count = min(10, len(y_pred))
    if recent_predictions_count > 0:
        result["recent_predictions"] = []
        for i in range(-recent_predictions_count, 0):
            result["recent_predictions"].append({
                "date": convert_numpy_to_json_serializable(valid_test_dates.iloc[i]),
                "actual": convert_numpy_to_json_serializable(testY_original[i]),
                "predicted": convert_numpy_to_json_serializable(y_pred[i]),
                "error": convert_numpy_to_json_serializable(abs(y_pred[i] - testY_original[i]))
            })

    result['is_new_model'] = is_new_model
    return result

# ============================================================================
# 멀티 실험 자동화 함수
# ============================================================================

def run_multiple_experiments(config_file="experiments.json"):
    """
    여러 실험을 순차적으로 실행하는 자동화 함수
    
    Args:
        config_file (str): 실험 설정 파일 경로
    
    Returns:
        list: 각 실험의 결과 리스트
    
    기능:
        1. JSON 설정 파일에서 실험 목록 로드
        2. 각 실험을 순차적으로 실행
        3. 실험 결과 요약 및 순위 생성
        4. 종합 결과를 JSON 파일로 저장
    
    생성 파일:
        - comprehensive_experiment_results.json: 전체 실험 결과 종합
        - predictions/{model}_predictions.json: 각 모델별 예측 결과
        - graphImage/{model}_*.png: 각 모델별 그래프
        - saved_models/{model}.*: 각 모델별 저장 파일
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
    
    # 총 실행 시간 계산
    total_end_time = time.time()
    total_time = round(total_end_time - total_start_time, 2)
    
    # ========================================================================
    # 결과 요약
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"📊 실험 결과 요약")
    print(f"{'='*60}")
    print(f"⏱️  총 실행 시간: {total_time}초")
    print(f"✅ 성공: {len([r for r in results if r['status'] == 'success'])}개")
    print(f"❌ 실패: {len([r for r in results if r['status'] == 'error'])}개")
    
    # 성공한 실험들의 정확도 순위
    successful_results = [r for r in results if r['status'] == 'success']
    if successful_results:
        successful_results.sort(key=lambda x: x['accuracy'], reverse=True)
        print(f"\n🏆 정확도 순위:")
        for i, result in enumerate(successful_results, 1):
            print(f"{i}. {result['experiment_name']}: {result['accuracy']:.2f}% (MAPE: {result['mape']:.2f}%)")
            print(f"   📈 R² Score: {result.get('r2_score', 'N/A')}, 방향성 정확도: {result.get('direction_accuracy', 'N/A'):.1f}%")
    
    # ========================================================================
    # 종합 결과 JSON 생성
    # ========================================================================
    comprehensive_results = {
        "experiment_summary": {
            "total_experiments": len(experiments),
            "successful_experiments": len(successful_results),
            "failed_experiments": len(results) - len(successful_results),
            "total_execution_time_seconds": total_time,
            "start_timestamp": datetime.now().isoformat(),
            "completion_timestamp": datetime.now().isoformat()
        },
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
        "detailed_results": results,
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
    
    # 종합 결과를 JSON 파일로 저장
    comprehensive_results_file = "comprehensive_experiment_results.json"
    with open(comprehensive_results_file, "w", encoding="utf-8") as f:
        json.dump(comprehensive_results, f, indent=2, ensure_ascii=False, default=convert_numpy_to_json_serializable)
    
    print(f"\n💾 종합 결과가 '{comprehensive_results_file}'에 저장되었습니다.")
    print(f"📁 개별 예측 결과는 'predictions/' 폴더에 저장되었습니다.")
    
    # 예측 파일 목록 출력
    if successful_results:
        print(f"\n📄 생성된 예측 파일 목록:")
        for result in successful_results:
            if 'prediction_file' in result:
                print(f"   - {result['prediction_file']}")
    
    return results

# ============================================================================
# 예측 파일 분석 함수
# ============================================================================

def analyze_prediction_file(prediction_file_path):
    """
    저장된 예측 파일을 분석하여 요약 정보 출력
    
    Args:
        prediction_file_path (str): 분석할 예측 파일 경로
    
    기능:
        - 모델 정보 출력 (모델명, 타겟 컬럼, 예측 개수)
        - 통계 정보 출력 (실제값/예측값 범위, MAE, RMSE)
        - 최근 5개 예측 결과 상세 출력
    """
    try:
        with open(prediction_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"\n📊 예측 파일 분석: {prediction_file_path}")
        print(f"{'='*50}")
        print(f"모델명: {data['model_name']}")
        print(f"타겟 컬럼: {data['target_column']}")
        print(f"예측 개수: {data['prediction_count']}")
        print(f"생성 시간: {data['timestamp']}")
        
        stats = data['statistics']
        print(f"\n📈 통계 정보:")
        print(f"   실제값 범위: {stats['actual_min']:.3f} ~ {stats['actual_max']:.3f}")
        print(f"   예측값 범위: {stats['predicted_min']:.3f} ~ {stats['predicted_max']:.3f}")
        print(f"   평균 절대 오차 (MAE): {stats['mean_absolute_error']:.4f}")
        print(f"   제곱근 평균 제곱 오차 (RMSE): {stats['rmse']:.4f}")
        
        # 최근 5개 예측 결과 출력
        predictions = data['predictions']
        print(f"\n🔍 최근 5개 예측 결과:")
        for pred in predictions[-5:]:
            error_pct = pred['percentage_error']
            print(f"   {pred['date'][:19]}: 실제={pred['actual_value']:.3f}, 예측={pred['predicted_value']:.3f}, 오차={error_pct:.2f}%")
            
    except Exception as e:
        print(f"❌ 예측 파일 분석 중 오류: {str(e)}")

# ============================================================================
# 데이터베이스 조회 함수들
# ============================================================================

def get_model_history(model_name=None, limit=10):
    """
    모델별 실험 이력 조회
    
    Args:
        model_name (str): 조회할 모델명 (None이면 전체 조회)
        limit (int): 조회할 최대 개수
    
    Returns:
        DataFrame: 실험 이력 데이터
            - model_name: 모델명
            - experiment_name: 실험명
            - accuracy, mape, r2_score: 성능 지표
            - created_at: 생성 일시
    """
    try:
        engine = get_db_engine()
        
        if model_name:
            # 특정 모델의 이력 조회
            query = f"""
            SELECT 
                m.model_name,
                m.epochs as model_epochs,
                m.sequence_length,
                e.experiment_id,
                e.experiment_name,
                e.accuracy,
                e.mape,
                e.r2_score,
                e.created_at
            FROM carbontwin.lstm_experiment e
            JOIN carbontwin.lstm_model m ON e.model_id = m.model_id
            WHERE m.model_name = '{model_name}'
            ORDER BY e.created_at DESC
            LIMIT {limit}
            """
        else:
            # 전체 모델 이력 조회
            query = f"""
            SELECT 
                m.model_name,
                m.epochs as model_epochs,
                e.experiment_id,
                e.experiment_name,
                e.accuracy,
                e.mape,
                e.r2_score,
                e.created_at
            FROM carbontwin.lstm_experiment e
            JOIN carbontwin.lstm_model m ON e.model_id = m.model_id
            ORDER BY e.created_at DESC
            LIMIT {limit}
            """
        
        return pd.read_sql_query(query, engine)
        
    except Exception as e:
        print(f"❌ 조회 오류: {str(e)}")
        return None

def get_best_models(metric='accuracy', top_n=5):
    """
    최고 성능 실험 조회
    
    Args:
        metric (str): 정렬 기준 (accuracy/mape/rmse/r2_score)
        top_n (int): 조회할 상위 개수
    
    Returns:
        DataFrame: 최고 성능 모델 데이터
            - model_name: 모델명
            - experiment_name: 실험명
            - 성능 지표들
            - 파일 경로들
    
    정렬 방식:
        - accuracy, r2_score: 내림차순 (높을수록 좋음)
        - mape, rmse: 오름차순 (낮을수록 좋음)
    """
    try:
        engine = get_db_engine()
        # 지표에 따라 정렬 순서 결정
        order = 'ASC' if metric in ['mape', 'rmse'] else 'DESC'
        
        query = f"""
        SELECT 
            m.model_name,
            m.epochs,
            m.sequence_length,
            m.prediction_days,
            e.experiment_name,
            e.accuracy,
            e.mape,
            e.rmse,
            e.r2_score,
            e.model_file_path,
            e.prediction_file_path,
            e.created_at
        FROM carbontwin.lstm_experiment e
        JOIN carbontwin.lstm_model m ON e.model_id = m.model_id
        WHERE e.status = 'success'
        ORDER BY e.{metric} {order}
        LIMIT {top_n}
        """
        
        return pd.read_sql_query(query, engine)
        
    except Exception as e:
        print(f"❌ 조회 오류: {str(e)}")
        return None

# ============================================================================
# 메인 실행부
# ============================================================================

if __name__ == "__main__":
    """
    프로그램 시작점
    
    실행 모드:
        1. 멀티 실험 (JSON 파일 기반)
        2. 단일 실험 (수동 입력)
        3. 예측 파일 분석
        4. DB에서 모델 이력 조회
        5. 최고 성능 모델 조회
    
    특징:
        - 07:00~16:45 시간대 데이터만 사용
        - 80/20 데이터 분할
        - 전체 예측 결과 상세 출력
        - PostgreSQL DB 연동
    """
    print("🧪 LSTM 멀티 실험 자동화 시스템 (예측값 기록 + DB 저장)")
    print("=" * 60)
    print("📋 이 시스템의 기능:")
    print("   1. 여러 LSTM 모델을 자동으로 학습 및 평가")
    print("   2. 예측 결과를 상세한 JSON 파일로 저장")
    print("   3. 성능 지표별 모델 순위 자동 생성")
    print("   4. 시각화 그래프 및 분석 리포트 생성")
    print("   5. PostgreSQL DB에 실험 결과 자동 저장")
    print("   6. 07:00~16:45 시간대 데이터만 사용 (80/20 split)")
    print("=" * 60)
    
    # 실행 모드 선택
    choice = input("실행 모드를 선택하세요:\n"
                  "1. 멀티 실험 (JSON 파일 기반)\n"
                  "2. 단일 실험 (수동 입력)\n"
                  "3. 예측 파일 분석\n"
                  "4. DB에서 모델 이력 조회\n"
                  "5. 최고 성능 모델 조회\n"
                  "선택 (1-5): ").strip()
    
    if choice == "1":
        # 멀티 실험 모드
        print("\n📖 멀티 실험 모드 설명:")
        print("   - experiments.json 파일의 설정에 따라 여러 실험을 순차 실행")
        print("   - 각 실험별로 모델 학습, 예측, 성능 평가를 자동화")
        print("   - 결과를 종합하여 성능 순위표 자동 생성")
        
        config_file = input("설정 파일명 (기본값: experiments.json): ").strip() or "experiments.json"
        results = run_multiple_experiments(config_file)
        
        if results and any(r['status'] == 'success' for r in results):
            print(f"\n🎉 모든 실험이 완료되었습니다!")
            print(f"📁 다음 파일들이 생성되었습니다:")
            print(f"   - comprehensive_experiment_results.json (종합 결과)")
            print(f"   - predictions/ 폴더 (개별 예측 파일들)")
            print(f"   - graphImage/ 폴더 (시각화 그래프들)")
            print(f"   - saved_models/ 폴더 (학습된 모델들)")
            
    elif choice == "2":
        # 단일 실험 모드
        print("\n📖 단일 실험 모드 안내:")
        print("   현재 단일 실험은 JSON 설정 파일을 통해 실행됩니다.")
        print("   experiments.json 파일을 생성하세요.")
        
    elif choice == "3":
        # 예측 파일 분석 모드
        print("\n📖 예측 파일 분석 모드")
        prediction_file = input("분석할 예측 파일 경로를 입력하세요: ").strip()
        if prediction_file and os.path.exists(prediction_file):
            analyze_prediction_file(prediction_file)
        else:
            print("❌ 파일을 찾을 수 없습니다.")
            # 사용 가능한 예측 파일 목록 출력
            if os.path.exists(prediction_path):
                pred_files = [f for f in os.listdir(prediction_path) if f.endswith('_predictions.json')]
                if pred_files:
                    print(f"\n📁 사용 가능한 예측 파일들:")
                    for i, file in enumerate(pred_files, 1):
                        print(f"   {i}. {file}")
                        
    elif choice == "4":
        # 모델 이력 조회 모드
        print("\n📊 모델 학습 이력 조회")
        model_name = input("모델명 입력 (전체: Enter): ").strip() or None
        limit = input("조회 개수 (기본 10개): ").strip() or "10"
        
        history = get_model_history(model_name, int(limit))
        if history is not None and not history.empty:
            print(f"\n📋 조회 결과 ({len(history)}개):")
            print(history.to_string(index=False))
        else:
            print("❌ 조회된 데이터가 없습니다.")
    
    elif choice == "5":
        # 최고 성능 모델 조회 모드
        print("\n🏆 최고 성능 모델 조회")
        metric = input("정렬 기준 (accuracy/mape/r2_score/rmse): ").strip() or "accuracy"
        top_n = input("조회 개수 (기본 5개): ").strip() or "5"

        best_models = get_best_models(metric, int(top_n))
        if best_models is not None and not best_models.empty:
            print(f"\n🎯 {metric} 기준 상위 {top_n}개:")
            print(best_models.to_string(index=False))
        else:
            print("❌ 조회된 데이터가 없습니다.")
    else:
        print("❌ 잘못된 선택입니다.")