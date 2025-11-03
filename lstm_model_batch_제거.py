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
        ORDER BY {dateColumn} ASC
        """
        
        data = pd.read_sql_query(query, engine)
        print(f"✅ 데이터 로드 성공: {len(data)}행 (07:00~16:45 데이터만)")
        
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
    """LSTM 모델 학습 (설정 객체 기반, 예측값 저장 포함)"""
    
    if not tf.executing_eagerly():
        tf.config.run_functions_eagerly(True)

    # 설정에서 파라미터 추출
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
    model_file_path = os.path.join(model_path, f"{modelName}.h5")

    stock_data = lstmData
    
    # 데이터 검증
    if stock_data.empty:
        return {"status": "error", "message": "데이터가 비어있습니다."}
    
    print(f"\n📊 로드된 데이터 정보:")
    print(f"   - 총 데이터 수: {len(stock_data)}개")
    
    study_columns_list = [col.strip() for col in studyColumns.split(',')]
    if targetColumn not in study_columns_list:
        return {"status": "error", "message": f"타겟 컬럼 '{targetColumn}'이 학습 컬럼에 없습니다."}

    # 날짜 컬럼 처리
    if dateColumn in stock_data.columns:
        dates = pd.to_datetime(stock_data[dateColumn], errors='coerce')
        
        # ✅ 시간 범위 확인
        hours = dates.dt.hour
        print(f"   - 시간 범위: {hours.min()}시 ~ {hours.max()}시")
        print(f"   - 고유 시간대: {sorted(hours.unique())}")
    else:
        dates = pd.date_range(start='2023-01-01', periods=len(stock_data), freq='15T')
        print(f"⚠️ 날짜 컬럼 '{dateColumn}'이 없어서 가상 날짜를 생성했습니다.")
    
    original_open = stock_data[targetColumn].values
    stock_data_for_training = stock_data[study_columns_list].astype(float)

    # 데이터 스케일링
    scaler = StandardScaler()
    stock_data_scaled = scaler.fit_transform(stock_data_for_training)

    # ✅ 80/20 split (기존 90/10에서 변경)
    split_index = int(len(stock_data_scaled) * 0.8)
    train_data_scaled = stock_data_scaled[:split_index]
    test_data_scaled = stock_data_scaled[split_index:]
    test_dates = dates[split_index:]

    pred_days = int(r_predDays)
    seq_len = int(r_seqLen)
    input_dim = stock_data_for_training.shape[1]
    target_idx = study_columns_list.index(targetColumn)

    # ✅ 데이터 충분성 검증
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

    # 시퀀스 데이터 생성
    trainX, trainY, testX, testY = [], [], [], []
    
    train_range = range(seq_len, len(train_data_scaled) - pred_days + 1)
    test_range = range(seq_len, len(test_data_scaled) - pred_days + 1)
    
    print(f"\n📊 시퀀스 생성 범위:")
    print(f"   - 학습 시퀀스: {len(train_range)}개")
    print(f"   - 테스트 시퀀스: {len(test_range)}개")
    
    if len(train_range) == 0:
        return {"status": "error", "message": "학습 시퀀스를 생성할 수 없습니다."}
    
    if len(test_range) == 0:
        return {"status": "error", "message": "테스트 시퀀스를 생성할 수 없습니다."}
    
    for i in train_range:
        trainX.append(train_data_scaled[i - seq_len:i, 0:input_dim])
        trainY.append(train_data_scaled[i + pred_days - 1:i + pred_days, target_idx])

    for i in test_range:
        testX.append(test_data_scaled[i - seq_len:i, 0:input_dim])
        testY.append(test_data_scaled[i + pred_days - 1:i + pred_days, target_idx])

    trainX, trainY = np.array(trainX), np.array(trainY)
    testX, testY = np.array(testX), np.array(testY)

    print(f"✅ 시퀀스 생성 완료:")
    print(f"   - trainX: {trainX.shape}, trainY: {trainY.shape}")
    print(f"   - testX: {testX.shape}, testY: {testY.shape}")

    print(f"\n🔄 {modelName} 모델 학습 시작...")
    is_new_model = False

    # 모델 생성 또는 로드
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
            def __init__(self, total_epochs, batch_size):
                super().__init__()
                self.total_epochs = total_epochs
                self.batch_size = batch_size
                self.prev_val_loss = None
                
            def on_train_begin(self, logs=None):
                print(f"🚀 모델 학습 시작 - 총 {self.total_epochs} 에포크")
                print(f"📊 배치 크기: {self.batch_size}")
                
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

        history = model.fit(
            trainX, trainY,
            epochs=int(r_epochs),
            batch_size=int(r_batchSize),
            validation_split=float(r_validationSplit),
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
        plt.title(f'{modelName} - Training Loss')
        plt.legend()
        plt.savefig(training_loss_path)
        plt.close()

    # 예측 수행
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

    # 예측 결과 역변환
    mean_values_pred = np.repeat(scaler.mean_[np.newaxis, :], prediction.shape[0], axis=0)
    mean_values_pred[:, target_idx] = np.squeeze(prediction)
    y_pred = scaler.inverse_transform(mean_values_pred)[:, target_idx]

    mean_values_testY = np.repeat(scaler.mean_[np.newaxis, :], testY.shape[0], axis=0)
    mean_values_testY[:, target_idx] = np.squeeze(testY)
    testY_original = scaler.inverse_transform(mean_values_testY)[:, target_idx]
    valid_test_dates = test_dates[seq_len : seq_len + len(testY_original)]

    # 예측 결과를 JSON으로 저장
    print(f"\n💾 예측 결과를 JSON 파일로 저장 중...")
    prediction_summary = save_predictions_to_json(
        modelName, 
        valid_test_dates, 
        testY_original, 
        y_pred, 
        targetColumn
    )

    # 전체 그래프 저장
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

    # 확대 그래프 저장
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

    # 정확도 계산
    print(f"\n📈 성능 평가 중...")
    
    # ✅ MAPE 함수 단순화 (DB에서 이미 시간 필터링됨)
    def mean_absolute_percentage_error(y_true, y_pred, valid_test_dates):
        print("valid_test_dates : ", valid_test_dates);
        eps = 9  # 임계값
        mask = y_true > eps
        
        print(f"\n📊 MAPE 계산 정보:")
        print(f"   - 임계값(eps): {eps}")
        print(f"   - 전체 데이터: {len(y_true)}개")
        print(f"   - 임계값 초과 데이터: {np.sum(mask)}개")
        
        if np.sum(mask) == 0:
            print("   ⚠️ 임계값을 초과하는 데이터가 없습니다.")
            return 999.0
        
        mape_value = np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100
        print(f"   - 계산된 MAPE: {mape_value:.2f}%")
        
        return mape_value

    try:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        sklearn_available = True
    except ImportError:
        print("⚠️ scikit-learn이 설치되지 않았습니다. 기본 지표만 계산합니다.")
        sklearn_available = False
    
    mape = mean_absolute_percentage_error(testY_original, y_pred, valid_test_dates)
    accuracy = 100 - mape if not np.isnan(mape) else 0
    
    # 추가 지표들
    if sklearn_available:
        mse = mean_squared_error(testY_original, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(testY_original, y_pred)
        r2 = r2_score(testY_original, y_pred)
    else:
        # 수동으로 계산
        mse = np.mean((testY_original - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(testY_original - y_pred))
        
        # R² 수동 계산
        ss_res = np.sum((testY_original - y_pred) ** 2)
        ss_tot = np.sum((testY_original - np.mean(testY_original)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # 방향성 정확도 (상승/하락 방향 예측 정확도)
    if len(testY_original) > 1:
        actual_direction = np.diff(testY_original) > 0
        pred_direction = np.diff(y_pred) > 0
        direction_accuracy = np.mean(actual_direction == pred_direction) * 100
    else:
        direction_accuracy = 0
    
    # 결과 출력
    print(f"\n📊 모델 성능 결과:")
    print(f"   🎯 MAPE: {mape:.2f}%")
    print(f"   📈 정확도: {accuracy:.2f}%")
    print(f"   📏 MAE: {mae:.4f}")
    print(f"   📐 RMSE: {rmse:.4f}")
    print(f"   🔍 R² Score: {r2:.4f}")
    print(f"   🧭 방향성 정확도: {direction_accuracy:.2f}%")
    
    # 성능 등급 계산
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
    
    # 과/소예측 분석
    over_predict = np.sum(y_pred > testY_original) / len(y_pred) * 100
    under_predict = 100 - over_predict
    print(f"   과예측 비율: {over_predict:.1f}%")
    print(f"   소예측 비율: {under_predict:.1f}%")

    # 설정 및 스케일러 저장
    with open(os.path.join(model_path, f"{modelName}_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    joblib.dump(scaler, os.path.join(model_path, f"{modelName}_scaler.pkl"))

    # 예측 데이터를 포함한 반환값
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
    
    # 최근 N개 예측값을 직접 결과에 포함
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

# ✅ 개별 예측 파일 분석 함수
def analyze_prediction_file(prediction_file_path):
    """저장된 예측 파일을 분석하여 요약 정보 출력"""
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
# 조회 함수
# ============================================================================
def get_model_history(model_name=None, limit=10):
    """모델별 실험 이력 조회"""
    try:
        engine = get_db_engine()
        
        if model_name:
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
    """최고 성능 실험 조회"""
    try:
        engine = get_db_engine()
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

# ✅ 메인 실행부
if __name__ == "__main__":
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
    
    choice = input("실행 모드를 선택하세요:\n"
                  "1. 멀티 실험 (JSON 파일 기반)\n"
                  "2. 단일 실험 (수동 입력)\n"
                  "3. 예측 파일 분석\n"
                  "4. DB에서 모델 이력 조회\n"
                  "5. 최고 성능 모델 조회\n"
                  "선택 (1-5): ").strip()
    
    if choice == "1":
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
        print("\n📖 단일 실험 모드 안내:")
        print("   현재 단일 실험은 JSON 설정 파일을 통해 실행됩니다.")
        print("   experiments.json 파일을 생성하세요.")
        
    elif choice == "3":
        print("\n📖 예측 파일 분석 모드")
        prediction_file = input("분석할 예측 파일 경로를 입력하세요: ").strip()
        if prediction_file and os.path.exists(prediction_file):
            analyze_prediction_file(prediction_file)
        else:
            print("❌ 파일을 찾을 수 없습니다.")
            if os.path.exists(prediction_path):
                pred_files = [f for f in os.listdir(prediction_path) if f.endswith('_predictions.json')]
                if pred_files:
                    print(f"\n📁 사용 가능한 예측 파일들:")
                    for i, file in enumerate(pred_files, 1):
                        print(f"   {i}. {file}")
                        
    elif choice == "4":
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