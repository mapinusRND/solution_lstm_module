# -*- coding: utf-8 -*-
"""
Title   : 외부데이터 기반 LSTM 예측 및 멀티 실험 자동화 모듈 (예측값 JSON 기록 기능 추가)
Author  : 주성중 / (주)맵인어스
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
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

# 환경 설정
ENV = os.getenv('FLASK_ENV', 'local')
if ENV == 'local':
    root = "D:/work/lstm"
else:
    root = "/app/webfiles/lstm"

# 경로 설정
graph_path = os.path.abspath(root + "/graphImage")
os.makedirs(graph_path, exist_ok=True)
model_path = os.path.abspath(root + "/saved_models")
os.makedirs(model_path, exist_ok=True)
# 예측 결과 저장 경로 추가
prediction_path = os.path.abspath(root + "/predictions")
os.makedirs(prediction_path, exist_ok=True)

# ✅ PostgreSQL 연결 함수 (SQLAlchemy 사용)
def get_db_engine():
    """SQLAlchemy 엔진 생성"""
    connection_string = "postgresql://postgres:mapinus@10.10.10.201:5432/postgres"
    # connection_string = "postgresql://postgres:7926@localhost:5432/postgres"
    return create_engine(connection_string)

# ✅ JSON 설정 파일 로드
def load_experiments_config(config_file="experiments.json"):
    """실험 설정 JSON 파일 로드"""
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

# ✅ 데이터 로드 함수 (07:00~16:45 필터링 추가)
def load_data_from_db(tablename, dateColumn, studyColumns):
    """데이터베이스에서 데이터 로드 (07:00~16:45만)"""
    try:
        engine = get_db_engine()
        # 
        query = f"""
        SELECT {studyColumns},{dateColumn}
        FROM carbontwin.{tablename}
        WHERE {dateColumn} IS NOT NULL
          AND TO_CHAR({dateColumn}, 'MM-DD') NOT IN (
            '06-01', '07-28', '07-29', '07-30', '07-31'
        )
        ORDER BY {dateColumn} ASC
        """
        # query = f"""
        # SELECT {studyColumns},{dateColumn}
        # FROM carbontwin.{tablename}
        # WHERE {dateColumn} IS NOT NULL
        #   AND TO_CHAR({dateColumn}, 'MM-DD') NOT IN (
        #     '06-01', '07-28', '07-29', '07-30', '07-31'
        # )
        # ORDER BY {dateColumn} ASC
        # """
        
        data = pd.read_sql_query(query, engine)

        # ✅ 시간대 분포 확인
        if dateColumn in data.columns and len(data) > 0:
            data[dateColumn] = pd.to_datetime(data[dateColumn])
            hours = data[dateColumn].dt.hour
            print(f"   📊 시간 범위: {hours.min()}시 ~ {hours.max()}시")
            hour_counts = hours.value_counts().sort_index()
            print(f"   📊 시간대별 데이터 수:")
            for hour, count in hour_counts.items():
                print(f"      {hour:2d}시: {count:5d}개")
        
        return data
        
    except Exception as e:
        print(f"❌ 데이터베이스 오류: {str(e)}")
        return None

# ✅ NumPy 배열을 JSON 직렬화 가능한 형태로 변환하는 함수
def convert_numpy_to_json_serializable(obj):
    """NumPy 배열과 특수 타입을 JSON 직렬화 가능한 형태로 변환"""
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

# ✅ 예측 결과를 JSON 형태로 저장하는 함수
def save_predictions_to_json(modelName, dates, actual_values, predicted_values, target_column):
    """예측 결과를 JSON 파일로 저장"""
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
        
        prediction_file_path = os.path.join(prediction_path, f"{modelName}_predictions.json")
        
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
        
        with open(prediction_file_path, 'w', encoding='utf-8') as f:
            json.dump(prediction_summary, f, indent=2, ensure_ascii=False)
        
        print(f"💾 예측 결과가 저장되었습니다: {prediction_file_path}")
        return prediction_summary
        
    except Exception as e:
        print(f"❌ 예측 결과 저장 중 오류: {str(e)}")
        return None

# ============================================================================
# save_experiment_to_db 함수
# ============================================================================
def save_experiment_to_db(result, config, is_new_model):
    """실험 결과를 DB에 저장"""
    try:
        engine = get_db_engine()
        model_name = result.get('modelName')
        
        if is_new_model:
            check_query = f"SELECT model_id FROM carbontwin.lstm_model WHERE model_name = '{model_name}'"
            existing = pd.read_sql_query(check_query, engine)
            
            if existing.empty:
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
                
                df_model = pd.DataFrame([model_data])
                df_model.to_sql('lstm_model', engine, schema='carbontwin',
                              if_exists='append', index=False)
                print(f"✅ 신규 모델 등록: {model_name}")
            else:
                print(f"ℹ️  기존 모델 사용: {model_name}")
        
        query = f"SELECT model_id FROM carbontwin.lstm_model WHERE model_name = '{model_name}'"
        model_id = pd.read_sql_query(query, engine).iloc[0]['model_id']
        
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

def run_single_experiment(experiment_config, experiment_index):
    """단일 실험 실행 및 DB 저장"""
    print(f"\n{'='*60}")
    print(f"🚀 실험 {experiment_index + 1} 시작: {experiment_config['name']}")
    print(f"{'='*60}")
    
    # 데이터 로드
    data = load_data_from_db(
        experiment_config['tablename'],
        experiment_config['dateColumn'], 
        experiment_config['studyColumns']
    )
    
    if data is None:
        return {"status": "error", "message": "데이터 로드 실패"}
    
    # 학습 실행
    start_time = time.time()
    result = lstmFinance(data, experiment_config)
    end_time = time.time()
    
    result['execution_time'] = round(end_time - start_time, 2)
    result['experiment_name'] = experiment_config['name']
    
    print(f"⏱️  실험 완료 시간: {result['execution_time']}초")
    
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

# ✅ LSTM 학습 함수 (시간 필터링 반영)
def lstmFinance(lstmData, config):
    """
    LSTM 모델 학습 - 요일별 패턴 고려 버전
    
    주요 개선 사항:
    - ✅ 평일/휴일 패턴 자동 학습
    - ✅ 요일별 샘플 가중치 적용
    - ✅ 평일/휴일별 성능 분석
    - ✅ 시간대별 특성 제거 (요일만 고려)
    """
    
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

    # 파일 경로 설정
    training_loss_path = os.path.join(graph_path, f"{modelName}_trainingLoss.png")
    total_graph_path = os.path.join(graph_path, f"{modelName}_totalgraph.png")
    diff_graph_path = os.path.join(graph_path, f"{modelName}_diffgraph.png")
    weekday_comparison_path = os.path.join(graph_path, f"{modelName}_weekday_comparison.png")
    model_file_path = os.path.join(model_path, f"{modelName}.h5")

    stock_data = lstmData
    
    # ====================================================================
    # 2단계: 데이터 검증
    # ====================================================================
    if stock_data.empty:
        return {"status": "error", "message": "데이터가 비어있습니다."}
    
    print(f"\n📊 로드된 데이터 정보:")
    print(f"   - 총 데이터 수: {len(stock_data)}개")
    
    study_columns_list = [col.strip() for col in studyColumns.split(',')]
    if targetColumn not in study_columns_list:
        return {"status": "error", "message": f"타겟 컬럼 '{targetColumn}'이 학습 컬럼에 없습니다."}

    # ====================================================================
    # 3단계: 날짜 컬럼 처리
    # ====================================================================
    if dateColumn in stock_data.columns:
        dates = pd.to_datetime(stock_data[dateColumn], errors='coerce')
    else:
        dates = pd.date_range(start='2023-01-01', periods=len(stock_data), freq='15T')
        print(f"⚠️ 날짜 컬럼 '{dateColumn}'이 없어서 가상 날짜를 생성했습니다.")
    
    # ====================================================================
    # 4단계: 평일/휴일 패턴 분석
    # ====================================================================
    print(f"\n🔍 요일별 데이터 패턴 분석...")
    
    target_values = stock_data[targetColumn].values
    weekday_names = ['월', '화', '수', '목', '금', '토', '일']
    
    # 요일별 통계 계산
    weekday_stats = {}
    for day_idx in range(7):
        day_mask = dates.dt.weekday == day_idx
        day_values = target_values[day_mask]
        
        if len(day_values) > 0:
            weekday_stats[day_idx] = {
                "name": weekday_names[day_idx],
                "mean": float(np.mean(day_values)),
                "std": float(np.std(day_values)),
                "median": float(np.median(day_values)),
                "count": len(day_values),
                "zero_ratio": float(np.sum(day_values == 0) / len(day_values)),
                "is_workday": day_idx < 5  # 월~금 = 평일
            }
    
    # 평일/휴일 그룹 통계
    workday_mask = dates.dt.weekday < 5
    holiday_mask = dates.dt.weekday >= 5
    
    workday_values = target_values[workday_mask]
    holiday_values = target_values[holiday_mask]
    
    workday_mean = np.mean(workday_values)
    workday_std = np.std(workday_values)
    holiday_mean = np.mean(holiday_values)
    holiday_std = np.std(holiday_values)
    
    print(f"\n   📊 평일/휴일 패턴:")
    print(f"      🏢 평일 (월~금): 평균 {workday_mean:.2f} (±{workday_std:.2f}), "
          f"데이터 {len(workday_values):,}개")
    print(f"      🏖️ 휴일 (토, 일): 평균 {holiday_mean:.2f} (±{holiday_std:.2f}), "
          f"데이터 {len(holiday_values):,}개")
    
    print(f"\n   📅 요일별 상세:")
    for day_idx in range(7):
        if day_idx in weekday_stats:
            stats = weekday_stats[day_idx]
            icon = "🏢" if stats["is_workday"] else "🏖️"
            print(f"      {icon} {stats['name']}요일: {stats['mean']:6.2f} kWh "
                  f"(±{stats['std']:5.2f}) | 0값: {stats['zero_ratio']*100:4.1f}% | "
                  f"데이터: {stats['count']:,}개")
    
    # 설정에 패턴 정보 추가 (나중에 예측 시 사용)
    config['weekday_patterns'] = {
        'workday': {'mean': workday_mean, 'std': workday_std},
        'holiday': {'mean': holiday_mean, 'std': holiday_std},
        'details': weekday_stats
    }
    
    # ====================================================================
    # 5단계: 데이터 준비 및 스케일링
    # ====================================================================
    original_open = stock_data[targetColumn].values
    stock_data_for_training = stock_data[study_columns_list].astype(float)

    # 데이터 스케일링
    scaler = StandardScaler()
    stock_data_scaled = scaler.fit_transform(stock_data_for_training)

    # 80/20 split
    split_index = int(len(stock_data_scaled) * 0.8)
    train_data_scaled = stock_data_scaled[:split_index]
    test_data_scaled = stock_data_scaled[split_index:]
    train_dates = dates[:split_index]
    test_dates = dates[split_index:]

    pred_days = int(r_predDays)
    seq_len = int(r_seqLen)
    input_dim = stock_data_for_training.shape[1]
    target_idx = study_columns_list.index(targetColumn)

    # ====================================================================
    # 6단계: 데이터 충분성 검증
    # ====================================================================
    print(f"\n🔍 시퀀스 생성 검증:")
    print(f"   - 전체 데이터: {len(stock_data_scaled)}개")
    print(f"   - 학습 데이터: {len(train_data_scaled)}개")
    print(f"   - 테스트 데이터: {len(test_data_scaled)}개")
    print(f"   - 시퀀스 길이(seq_len): {seq_len}")
    print(f"   - 예측 일수(pred_days): {pred_days}")
    
    min_required = seq_len + pred_days
    print(f"   - 필요한 최소 데이터: {min_required}개")
    
    if len(train_data_scaled) < min_required:
        error_msg = f"학습 데이터 부족: {len(train_data_scaled)}개 (최소 {min_required}개 필요)"
        print(f"❌ {error_msg}")
        return {"status": "error", "message": error_msg}
    
    if len(test_data_scaled) < min_required:
        error_msg = f"테스트 데이터 부족: {len(test_data_scaled)}개 (최소 {min_required}개 필요)"
        print(f"❌ {error_msg}")
        return {"status": "error", "message": error_msg}

    # ====================================================================
    # 7단계: 시퀀스 데이터 생성 (샘플 가중치 포함)
    # ====================================================================
    trainX, trainY, train_sample_weights = [], [], []
    testX, testY, test_sample_dates = [], [], []
    
    train_range = range(seq_len, len(train_data_scaled) - pred_days + 1)
    test_range = range(seq_len, len(test_data_scaled) - pred_days + 1)
    
    print(f"\n📊 시퀀스 생성 범위:")
    print(f"   - 학습 시퀀스: {len(train_range)}개")
    print(f"   - 테스트 시퀀스: {len(test_range)}개")
    
    if len(train_range) == 0:
        return {"status": "error", "message": "학습 시퀀스를 생성할 수 없습니다."}
    
    if len(test_range) == 0:
        return {"status": "error", "message": "테스트 시퀀스를 생성할 수 없습니다."}
    
    # ====================================================================
    # 학습 데이터 생성 + 샘플 가중치 계산
    # ====================================================================
    print(f"\n⚖️ 샘플 가중치 계산 중...")
    
    # 평일/휴일 비율 계산
    workday_count = np.sum(train_dates.dt.weekday < 5)
    holiday_count = len(train_dates) - workday_count
    total_count = len(train_dates)
    
    # 역비율 가중치 (데이터가 적은 그룹에 높은 가중치)
    workday_weight = total_count / (2 * workday_count) if workday_count > 0 else 1.0
    holiday_weight = total_count / (2 * holiday_count) if holiday_count > 0 else 1.0
    
    print(f"   - 평일 데이터: {workday_count:,}개 (가중치: {workday_weight:.3f})")
    print(f"   - 휴일 데이터: {holiday_count:,}개 (가중치: {holiday_weight:.3f})")
    
    for i in train_range:
        trainX.append(train_data_scaled[i - seq_len:i, 0:input_dim])
        trainY.append(train_data_scaled[i + pred_days - 1:i + pred_days, target_idx])
        
        # 타겟 날짜의 요일에 따라 가중치 부여
        target_date_idx = i + pred_days - 1
        if target_date_idx < len(train_dates):
            target_weekday = train_dates.iloc[target_date_idx].weekday()
            # 평일(0~4) vs 휴일(5~6)
            if target_weekday < 5:  # 평일
                train_sample_weights.append(workday_weight)
            else:  # 휴일
                train_sample_weights.append(holiday_weight)
        else:
            train_sample_weights.append(1.0)
    
    # 테스트 데이터 생성
    for i in test_range:
        testX.append(test_data_scaled[i - seq_len:i, 0:input_dim])
        testY.append(test_data_scaled[i + pred_days - 1:i + pred_days, target_idx])
        
        target_date_idx = i + pred_days - 1
        if target_date_idx < len(test_dates):
            test_sample_dates.append(test_dates.iloc[target_date_idx])
        else:
            test_sample_dates.append(test_dates.iloc[-1])

    trainX, trainY = np.array(trainX), np.array(trainY)
    train_sample_weights = np.array(train_sample_weights)
    testX, testY = np.array(testX), np.array(testY)

    print(f"✅ 시퀀스 생성 완료:")
    print(f"   - trainX: {trainX.shape}, trainY: {trainY.shape}")
    print(f"   - testX: {testX.shape}, testY: {testY.shape}")
    print(f"   - 샘플 가중치: {train_sample_weights.shape}")

    # ====================================================================
    # 8단계: 모델 생성 또는 로드
    # ====================================================================
    print(f"\n🔄 {modelName} 모델 학습 시작...")
    is_new_model = False

    try:
        model = load_model(model_file_path, compile=False)
        model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
        print("✅ 기존 모델 로드됨")
        is_new_model = False
    except (OSError, IOError):
        print("🔄 새 모델 생성 중...")
        is_new_model = True

        model = Sequential([
            Input(shape=(trainX.shape[1], trainX.shape[2])),
            LSTM(64, return_sequences=True),
            LSTM(32, return_sequences=False),
            Dense(trainY.shape[1])
        ])

        model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')

        class TrainingCallback(Callback):
            """학습 과정을 모니터링하는 콜백"""
            def __init__(self, total_epochs, batch_size):
                super().__init__()
                self.total_epochs = total_epochs
                self.batch_size = batch_size
                self.prev_val_loss = None
                
            def on_train_begin(self, logs=None):
                print(f"🚀 모델 학습 시작 - 총 {self.total_epochs} 에포크")
                print(f"📊 배치 크기: {self.batch_size}")
                print(f"⚖️ 샘플 가중치 적용: 평일/휴일 균형 학습")
                
            def on_epoch_begin(self, epoch, logs=None):
                print(f"\n⏳ Epoch {epoch + 1}/{self.total_epochs} 시작...")
                
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                loss = logs.get('loss', 0)
                val_loss = logs.get('val_loss', 0)
                
                progress = (epoch + 1) / self.total_epochs * 100
                bar_length = 30
                filled_length = int(bar_length * (epoch + 1) // self.total_epochs)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                
                print(f"✅ Epoch {epoch + 1}/{self.total_epochs} [{bar}] {progress:.1f}%")
                print(f"   📉 Loss: {loss:.6f} | Val_Loss: {val_loss:.6f}")
                
                if epoch > 0 and self.prev_val_loss is not None:
                    if val_loss < self.prev_val_loss:
                        print(f"   📈 검증 손실 개선: {self.prev_val_loss:.6f} → {val_loss:.6f}")
                    elif val_loss > self.prev_val_loss * 1.1:
                        print(f"   ⚠️  검증 손실 증가: {self.prev_val_loss:.6f} → {val_loss:.6f}")
                
                self.prev_val_loss = val_loss
                
            def on_train_end(self, logs=None):
                print(f"\n🎉 학습 완료!")

        # ✅ 샘플 가중치를 적용하여 학습
        print(f"\n⚖️ 평일/휴일 균형 학습 시작...")
        history = model.fit(
            trainX, trainY,
            epochs=int(r_epochs),
            batch_size=int(r_batchSize),
            validation_split=float(r_validationSplit),
            sample_weight=train_sample_weights,  # ✅ 샘플 가중치 적용!
            verbose=1,
            callbacks=[TrainingCallback(int(r_epochs), int(r_batchSize))]
        )

        model.save(model_file_path)
        print("✅ 모델 저장 완료")

        # 학습 손실 그래프 저장
        plt.figure(figsize=(12, 4))
        plt.plot(history.history['loss'], label='Training loss')
        plt.plot(history.history['val_loss'], label='Validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{modelName} - Training Loss (With Sample Weights)')
        plt.legend()
        plt.savefig(training_loss_path)
        plt.close()

    # ====================================================================
    # 9단계: 예측 수행
    # ====================================================================
    print(f"\n🔮 예측 수행 중...")
    print(f"📊 예측할 샘플 수: {len(testX)}")
    
    batch_size_pred = 32
    predictions = []
    total_batches = (len(testX) + batch_size_pred - 1) // batch_size_pred
    
    for i in range(0, len(testX), batch_size_pred):
        batch_end = min(i + batch_size_pred, len(testX))
        batch_data = testX[i:batch_end]
        
        batch_pred = model.predict(batch_data, verbose=0)
        predictions.append(batch_pred)
        
        current_batch = (i // batch_size_pred) + 1
        progress = current_batch / total_batches * 100
        bar_length = 25
        filled_length = int(bar_length * current_batch // total_batches)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        print(f"\r⏳ 예측 진행: [{bar}] {progress:.1f}% ({current_batch}/{total_batches} 배치)", end='', flush=True)
    
    prediction = np.vstack(predictions)
    print(f"\n✅ 예측 완료! 총 {len(prediction)}개 샘플 예측됨")

    # ====================================================================
    # 10단계: 예측 결과 역변환
    # ====================================================================
    mean_values_pred = np.repeat(scaler.mean_[np.newaxis, :], prediction.shape[0], axis=0)
    mean_values_pred[:, target_idx] = np.squeeze(prediction)
    y_pred = scaler.inverse_transform(mean_values_pred)[:, target_idx]

    mean_values_testY = np.repeat(scaler.mean_[np.newaxis, :], testY.shape[0], axis=0)
    mean_values_testY[:, target_idx] = np.squeeze(testY)
    testY_original = scaler.inverse_transform(mean_values_testY)[:, target_idx]
    
    valid_test_dates = test_dates[seq_len : seq_len + len(testY_original)]

    # ====================================================================
    # 11단계: 평일/휴일별 성능 분석
    # ====================================================================
    print(f"\n📊 평일/휴일별 성능 분석...")
    
    # 예측 날짜의 요일 확인
    test_weekdays = pd.Series([d.weekday() for d in test_sample_dates[:len(testY_original)]])
    workday_mask_test = test_weekdays < 5
    holiday_mask_test = test_weekdays >= 5
    
    # 평일 성능
    workday_actual = testY_original[workday_mask_test]
    workday_pred = y_pred[workday_mask_test]
    
    # 휴일 성능
    holiday_actual = testY_original[holiday_mask_test]
    holiday_pred = y_pred[holiday_mask_test]
    
    print(f"\n   🏢 평일 예측 성능:")
    if len(workday_actual) > 0:
        workday_mae = np.mean(np.abs(workday_pred - workday_actual))
        workday_rmse = np.sqrt(np.mean((workday_pred - workday_actual) ** 2))
        workday_mape = np.mean(np.abs((workday_pred - workday_actual) / workday_actual)) * 100
        print(f"      - 샘플 수: {len(workday_actual)}개")
        print(f"      - MAE: {workday_mae:.4f}")
        print(f"      - RMSE: {workday_rmse:.4f}")
        print(f"      - MAPE: {workday_mape:.2f}%")
        print(f"      - 실제 평균: {np.mean(workday_actual):.2f}")
        print(f"      - 예측 평균: {np.mean(workday_pred):.2f}")
    
    print(f"\n   🏖️ 휴일 예측 성능:")
    if len(holiday_actual) > 0:
        holiday_mae = np.mean(np.abs(holiday_pred - holiday_actual))
        holiday_rmse = np.sqrt(np.mean((holiday_pred - holiday_actual) ** 2))
        # 0 방지
        valid_holiday_mask = holiday_actual > 1
        if np.sum(valid_holiday_mask) > 0:
            holiday_mape = np.mean(np.abs((holiday_pred[valid_holiday_mask] - holiday_actual[valid_holiday_mask]) / holiday_actual[valid_holiday_mask])) * 100
        else:
            holiday_mape = 999.0
        
        print(f"      - 샘플 수: {len(holiday_actual)}개")
        print(f"      - MAE: {holiday_mae:.4f}")
        print(f"      - RMSE: {holiday_rmse:.4f}")
        print(f"      - MAPE: {holiday_mape:.2f}%")
        print(f"      - 실제 평균: {np.mean(holiday_actual):.2f}")
        print(f"      - 예측 평균: {np.mean(holiday_pred):.2f}")

    # ====================================================================
    # 12단계: 요일별 비교 그래프 생성
    # ====================================================================
    print(f"\n📊 요일별 비교 그래프 생성 중...")
    
    weekday_actual_means = []
    weekday_pred_means = []
    weekday_labels = []
    
    for day_idx in range(7):
        day_mask = test_weekdays == day_idx
        if np.sum(day_mask) > 0:
            day_actual = testY_original[day_mask]
            day_pred = y_pred[day_mask]
            weekday_actual_means.append(np.mean(day_actual))
            weekday_pred_means.append(np.mean(day_pred))
            weekday_labels.append(weekday_names[day_idx])
    
    if len(weekday_labels) > 0:
        plt.figure(figsize=(12, 6))
        x = np.arange(len(weekday_labels))
        width = 0.35
        
        plt.bar(x - width/2, weekday_actual_means, width, label='실제 평균', alpha=0.8)
        plt.bar(x + width/2, weekday_pred_means, width, label='예측 평균', alpha=0.8)
        
        plt.xlabel('요일')
        plt.ylabel(f'{targetColumn} 평균값')
        plt.title(f'{modelName} - 요일별 실제 vs 예측 비교')
        plt.xticks(x, weekday_labels)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(weekday_comparison_path)
        plt.close()
        print(f"✅ 요일별 비교 그래프 저장: {weekday_comparison_path}")

    # ====================================================================
    # 13단계: 예측 결과 JSON 저장
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
    # 14단계: 시각화 그래프 생성
    # ====================================================================
    # 전체 그래프
    plt.figure(figsize=(15, 5))
    plt.plot(dates, original_open, color='green', label=f'Original {targetColumn}', alpha=0.7)
    plt.plot(valid_test_dates, testY_original, color='blue', label=f'Actual {targetColumn}')
    plt.plot(valid_test_dates, y_pred, color='red', linestyle='--', label=f'Predicted {targetColumn}')
    plt.xlabel(dateColumn)
    plt.ylabel(f'{targetColumn} Value')
    plt.title(f'{modelName} - Prediction Results (Weekday-aware)')
    plt.legend()
    plt.savefig(total_graph_path)
    plt.close()

    # 확대 그래프
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
    # 15단계: 성능 지표 계산
    # ====================================================================
    print(f"\n📈 성능 평가 중...")
    
    def mean_absolute_percentage_error(y_true, y_pred, valid_test_dates, eps=9):
        """MAPE 계산 함수"""
        mask = y_true > eps
        
        print(f"\n📊 MAPE 계산 정보:")
        print(f"   - 임계값(eps): {eps}")
        print(f"   - 전체 데이터: {len(y_true)}개")
        print(f"   - 임계값 초과 데이터: {np.sum(mask)}개")
        
        if np.sum(mask) == 0:
            print("   ⚠️ 임계값을 초과하는 데이터가 없습니다.")
            return 999.0
        
        mape_value = np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100
        print(f"   ✅ 계산된 MAPE: {mape_value:.2f}%")
        
        return mape_value

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
    
    # 방향성 정확도
    if len(testY_original) > 1:
        actual_direction = np.diff(testY_original) > 0
        pred_direction = np.diff(y_pred) > 0
        direction_accuracy = np.mean(actual_direction == pred_direction) * 100
    else:
        direction_accuracy = 0
    
    # 결과 출력
    print(f"\n📊 전체 모델 성능:")
    print(f"   🎯 MAPE: {mape:.2f}%")
    print(f"   📈 정확도: {accuracy:.2f}%")
    print(f"   📏 MAE: {mae:.4f}")
    print(f"   📐 RMSE: {rmse:.4f}")
    print(f"   🔍 R² Score: {r2:.4f}")
    print(f"   🧭 방향성 정확도: {direction_accuracy:.2f}%")
    
    # 성능 등급
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
    
    over_predict = np.sum(y_pred > testY_original) / len(y_pred) * 100
    under_predict = 100 - over_predict
    print(f"   과예측 비율: {over_predict:.1f}%")
    print(f"   소예측 비율: {under_predict:.1f}%")

    # ====================================================================
    # 16단계: 설정 및 스케일러 저장
    # ====================================================================
    with open(os.path.join(model_path, f"{modelName}_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    joblib.dump(scaler, os.path.join(model_path, f"{modelName}_scaler.pkl"))

    # ====================================================================
    # 17단계: 결과 반환
    # ====================================================================
    result = {
        "status": "success",
        "modelName": modelName,
        "training_loss_img": f"graphImage/{modelName}_trainingLoss.png",
        "total_graph_img": f"graphImage/{modelName}_totalgraph.png",
        "diff_graph_img": f"graphImage/{modelName}_diffgraph.png",
        "weekday_comparison_img": f"graphImage/{modelName}_weekday_comparison.png",
        "mape": round(mape, 2),
        "accuracy": round(accuracy, 2),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2_score": round(r2, 4),
        "direction_accuracy": round(direction_accuracy, 2),
        "prediction_file": f"predictions/{modelName}_predictions.json",
        "weekday_performance": {
            "workday": {
                "count": int(len(workday_actual)) if len(workday_actual) > 0 else 0,
                "mae": round(float(workday_mae), 4) if len(workday_actual) > 0 else 0,
                "rmse": round(float(workday_rmse), 4) if len(workday_actual) > 0 else 0,
                "mape": round(float(workday_mape), 2) if len(workday_actual) > 0 else 999
            },
            "holiday": {
                "count": int(len(holiday_actual)) if len(holiday_actual) > 0 else 0,
                "mae": round(float(holiday_mae), 4) if len(holiday_actual) > 0 else 0,
                "rmse": round(float(holiday_rmse), 4) if len(holiday_actual) > 0 else 0,
                "mape": round(float(holiday_mape), 2) if len(holiday_actual) > 0 else 999
            }
        },
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
    
    # 최근 예측값 추가
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
    
    print(f"\n✅ 학습 및 평가 완료!")
    print(f"📊 평일/휴일 패턴 정보가 config에 저장되었습니다.")
    
    return result

# ✅ 멀티 실험 실행 함수
def run_multiple_experiments(config_file="experiments.json"):
    """여러 실험을 순차적으로 실행 (예측값 포함)"""
    experiments = load_experiments_config(config_file)
    
    if not experiments:
        print("❌ 실행할 실험이 없습니다.")
        return
    
    print(f"🔬 총 {len(experiments)}개의 실험을 시작합니다.")
    print(f"⏰ 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    total_start_time = time.time()
    
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
    
    # 결과 요약
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
    
    # 전체 실험 결과 및 예측 데이터를 포함한 종합 JSON 저장
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


# ✅ 메인 실행부
if __name__ == "__main__":
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
        
    