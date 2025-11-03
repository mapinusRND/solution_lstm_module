# -*- coding: utf-8 -*-
"""
Title   : 외부데이터 기반 LSTM 예측 및 멀티 실험 자동화 모듈
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

# ✅ PostgreSQL 연결 함수 (SQLAlchemy 사용)
def get_db_engine():
    """SQLAlchemy 엔진 생성"""
    connection_string = "postgresql://postgres:mapinus@10.10.10.201:5432/postgres"
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

# ✅ 데이터 로드 함수
def load_data_from_db(tablename, dateColumn, studyColumns):
    """데이터베이스에서 데이터 로드"""
    try:
        engine = get_db_engine()
        
        query = f"""
        SELECT {studyColumns},{dateColumn}
        FROM carbontwin.{tablename}
        WHERE {dateColumn} IS NOT NULL
        ORDER BY {dateColumn} ASC
        """
        
        data = pd.read_sql_query(query, engine)
        print(f"✅ 데이터 로드 성공: {len(data)}행")
        return data
        
    except Exception as e:
        print(f"❌ 데이터베이스 오류: {str(e)}")
        return None

# ✅ 단일 실험 실행 함수
def run_single_experiment(experiment_config, experiment_index):
    """단일 실험 실행"""
    print(f"\n{'='*60}")
    print(f"🚀 실험 {experiment_index + 1}/{len(experiment_config)} 시작: {experiment_config['name']}")
    print(f"{'='*60}")
    
    # 설정 출력
    print(f"📋 실험 설정:")
    print(f"   - 테이블: {experiment_config['tablename']}")
    print(f"   - 모델명: {experiment_config['modelName']}")
    print(f"   - 타겟 컬럼: {experiment_config['targetColumn']}")
    print(f"   - 에포크: {experiment_config['r_epochs']}")
    print(f"   - 배치크기: {experiment_config['r_batchSize']}")
    print(f"   - 시퀀스길이: {experiment_config['r_seqLen']}")
    
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
    
    # 실행 시간 추가
    result['execution_time'] = round(end_time - start_time, 2)
    result['experiment_name'] = experiment_config['name']
    
    print(f"⏱️  실험 완료 시간: {result['execution_time']}초")
    return result

# ✅ LSTM 학습 함수 (수정됨)
def lstmFinance(lstmData, config):
    """LSTM 모델 학습 (설정 객체 기반)"""
    
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
    
    study_columns_list = [col.strip() for col in studyColumns.split(',')]
    if targetColumn not in study_columns_list:
        return {"status": "error", "message": f"타겟 컬럼 '{targetColumn}'이 학습 컬럼에 없습니다."}

    # 날짜 컬럼 처리
    if dateColumn in stock_data.columns:
        dates = pd.to_datetime(stock_data[dateColumn], errors='coerce')
    else:
        dates = pd.date_range(start='2023-01-01', periods=len(stock_data), freq='5T')
        print(f"⚠️ 날짜 컬럼 '{dateColumn}'이 없어서 가상 날짜를 생성했습니다.")
    
    original_open = stock_data[targetColumn].values
    stock_data_for_training = stock_data[study_columns_list].astype(float)

    # 데이터 스케일링
    scaler = StandardScaler()
    stock_data_scaled = scaler.fit_transform(stock_data_for_training)

    n_train = int(0.9 * stock_data_scaled.shape[0])
    train_data_scaled = stock_data_scaled[:n_train]
    test_data_scaled = stock_data_scaled[n_train:]
    test_dates = dates[n_train:]

    pred_days = int(r_predDays)
    seq_len = int(r_seqLen)
    input_dim = stock_data_for_training.shape[1]
    target_idx = study_columns_list.index(targetColumn)

    # 시퀀스 데이터 생성
    trainX, trainY, testX, testY = [], [], [], []
    for i in range(seq_len, n_train - pred_days + 1):
        trainX.append(train_data_scaled[i - seq_len:i, 0:input_dim])
        trainY.append(train_data_scaled[i + pred_days - 1:i + pred_days, target_idx])

    for i in range(seq_len, len(test_data_scaled) - pred_days + 1):
        testX.append(test_data_scaled[i - seq_len:i, 0:input_dim])
        testY.append(test_data_scaled[i + pred_days - 1:i + pred_days, target_idx])

    trainX, trainY = np.array(trainX), np.array(trainY)
    testX, testY = np.array(testX), np.array(testY)

    print(f"🔄 {modelName} 모델 학습 시작...")
    print(f"📊 훈련 데이터: {trainX.shape}, 테스트 데이터: {testX.shape}")

    # 모델 생성 또는 로드
    try:
        model = load_model(model_file_path, compile=False)
        model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
        print("✅ 기존 모델 로드됨")
    except (OSError, IOError):
        print("🔄 새 모델 생성 중...")

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
                
                # 진행률 계산
                progress = (epoch + 1) / self.total_epochs * 100
                
                # 진행바 생성
                bar_length = 30
                filled_length = int(bar_length * (epoch + 1) // self.total_epochs)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                
                print(f"✅ Epoch {epoch + 1}/{self.total_epochs} [{bar}] {progress:.1f}%")
                print(f"   📉 Loss: {loss:.6f} | Val_Loss: {val_loss:.6f}")
                
                # 개선 여부 확인
                if epoch > 0 and self.prev_val_loss is not None:
                    if val_loss < self.prev_val_loss:
                        print(f"   📈 검증 손실 개선: {self.prev_val_loss:.6f} → {val_loss:.6f}")
                    elif val_loss > self.prev_val_loss * 1.1:  # 10% 이상 증가시 경고
                        print(f"   ⚠️  검증 손실 증가: {self.prev_val_loss:.6f} → {val_loss:.6f}")
                
                self.prev_val_loss = val_loss
                
            def on_train_end(self, logs=None):
                print(f"\n🎉 학습 완료!")

        history = model.fit(
            trainX, trainY,
            epochs=int(r_epochs),
            batch_size=int(r_batchSize),
            validation_split=float(r_validationSplit),
            verbose=1,  # 기본 진행상황 표시
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
    
    # 배치별로 예측하여 진행상황 표시
    batch_size_pred = 32  # 예측용 배치 크기
    predictions = []
    
    total_batches = (len(testX) + batch_size_pred - 1) // batch_size_pred
    
    for i in range(0, len(testX), batch_size_pred):
        batch_end = min(i + batch_size_pred, len(testX))
        batch_data = testX[i:batch_end]
        
        batch_pred = model.predict(batch_data, verbose=0)
        predictions.append(batch_pred)
        
        # 진행상황 표시
        current_batch = (i // batch_size_pred) + 1
        progress = current_batch / total_batches * 100
        
        # 진행바 생성
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
    
    def mean_absolute_percentage_error(y_true, y_pred):
        mask = y_true != 0
        if np.sum(mask) == 0:
            return 999.0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    # 추가 평가 지표들 계산
    try:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        sklearn_available = True
    except ImportError:
        print("⚠️ scikit-learn이 설치되지 않았습니다. 기본 지표만 계산합니다.")
        sklearn_available = False
    
    mape = mean_absolute_percentage_error(testY_original, y_pred)
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

    return {
        "status": "success",
        "modelName": modelName,
        "training_loss_img": f"graphImage/{modelName}_trainingLoss.png",
        "total_graph_img": f"graphImage/{modelName}_totalgraph.png",
        "diff_graph_img": f"graphImage/{modelName}_diffgraph.png",
        "mape": round(mape, 2),
        "accuracy": round(accuracy, 2)
    }

# ✅ 멀티 실험 실행 함수
def run_multiple_experiments(config_file="experiments.json"):
    """여러 실험을 순차적으로 실행"""
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
    
    # 결과를 JSON 파일로 저장
    with open("experiment_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_experiments": len(experiments),
                "successful": len(successful_results),
                "failed": len(results) - len(successful_results),
                "total_time": total_time,
                "timestamp": datetime.now().isoformat()
            },
            "results": results
        }, f, indent=4, ensure_ascii=False)
    
    print(f"\n💾 결과가 'experiment_results.json'에 저장되었습니다.")
    return results

# ✅ 메인 실행부
if __name__ == "__main__":
    print("🧪 LSTM 멀티 실험 자동화 시스템")
    print("=" * 50)
    
    choice = input("실행 모드를 선택하세요:\n1. 멀티 실험 (JSON 파일 기반)\n2. 단일 실험 (수동 입력)\n선택 (1 또는 2): ").strip()
    
    if choice == "1":
        config_file = input("설정 파일명 (기본값: experiments.json): ").strip() or "experiments.json"
        run_multiple_experiments(config_file)
    elif choice == "2":
        # 기존 단일 실험 모드
        print("단일 실험 모드는 기존 코드를 사용하세요.")
    else:
        print("잘못된 선택입니다.")