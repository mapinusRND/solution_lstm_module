# -*- coding: utf-8 -*-
"""
Title   : 외부데이터 기반 LSTM 예측 및 실시간 학습
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
from sqlalchemy import create_engine  # ✅ SQLAlchemy 추가

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

def get_db_connection():
    """기존 psycopg2 연결 (필요시 사용)"""
    return psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="mapinus",
        host="10.10.10.201",
        port="5432"
    )

# 🔸 주식 데이터 기반 LSTM 모델 학습 API
def lstmLearningStock():
    """
    Flask 애플리케이션 컨텍스트 없이 실행 가능한 버전
    """
    # 루트 디렉토리 및 저장 경로 설정
    makeRoot = os.getenv("ROOT_PATH", root)
    finance_path = os.path.join(makeRoot, "finance_data")
    os.makedirs(finance_path, exist_ok=True)

    # ✅ 추가된 컬럼 관련 파라미터 수신
    dateColumn = "time_point"
    studyColumns = "solar_kwh,usage_kwh"
    targetColumn = "solar_kwh"

    lstmData = None

    # DB에서 데이터를 뽑아서 data 변수에 담기
    try:
        tablename = "lstm_input_5m"
        engine = get_db_engine()  # ✅ SQLAlchemy 엔진 사용

        # ✅ AWS 관측 정보 + 위치 정보 조인 쿼리
        query = f"""
        SELECT {studyColumns},{dateColumn}
        FROM carbontwin.{tablename}
        WHERE {dateColumn} IS NOT NULL
        ORDER BY {dateColumn} ASC
        """

        # ✅ SQLAlchemy 엔진을 사용하여 경고 해결
        lstmData = pd.read_sql_query(query, engine)
        print(f"✅ 데이터 로드 성공: {len(lstmData)}행")
        
    except Exception as e:
        print(f"❌ 데이터베이스 오류: {str(e)}")
        return {"status": "error", "message": str(e)}
        
    # 🔹 백그라운드 작업으로 학습 함수 실행
    def background_task():
        try:
            result = lstmFinance(lstmData, dateColumn, studyColumns, targetColumn)
            print("✅ 백그라운드 작업 성공:", result)
            return result
        except Exception as e:
            print("❌ 백그라운드 작업 오류:", str(e))
            return {"status": "error", "message": str(e)}

    # ✅ 실제로 백그라운드 작업 실행
    result = background_task()
    
    return {"status": "success", "message": "학습이 시작되었습니다.", "result": result}

# 메인 함수
def lstmFinance(lstmData, dateColumn, studyColumns, targetColumn):

    if not tf.executing_eagerly():
        tf.config.run_functions_eagerly(True)

    # modelName = "training_1"
    try:
        modelName = input("모델명: ").strip() or "training_1"
        r_epochs = int(input("에포크 수 (기본값: 20): ").strip() or "20")
        r_batchSize = int(input("배치 크기 (기본값: 16): ").strip() or "16")
        r_validationSplit = float(input("검증 데이터 비율 (기본값: 0.1): ").strip() or "0.1")
        r_seqLen = int(input("시퀀스 길이 (기본값: 14): ").strip() or "14")
        r_predDays = int(input("예측 일수 (기본값: 1): ").strip() or "1")
    except ValueError:
        print("❌ 잘못된 입력값입니다. 기본값을 사용합니다.")
        modelName = "training_1"
        r_epochs = 20
        r_batchSize = 16
        r_validationSplit = 0.1
        r_seqLen = 14
        r_predDays = 1
    # r_epochs = 20
    # r_batchSize = 16
    # r_validationSplit = 0.1
    # r_seqLen = 14
    # r_predDays = 1
    sessionId = "train_user_1"

    training_loss_path = graph_path + "/" + modelName + "_trainingLoss.png"
    total_graph_path = graph_path + "/" + modelName + "_totalgraph.png"
    diff_graph_path = graph_path + "/" + modelName + "_diffgraph.png"
    model_file_path = os.path.join(model_path, modelName + ".h5")

    stock_data = lstmData
    
    # ✅ 데이터 검증 추가
    if stock_data.empty:
        print(f"❌ 데이터가 비어있습니다.")
        return {"status": "error", "message": "데이터가 비어있습니다."}
    
    # ✅ targetColumn이 studyColumns에 있는지 확인
    study_columns_list = [col.strip() for col in studyColumns.split(',')]
    if targetColumn not in study_columns_list:
        print(f"❌ 타겟 컬럼 '{targetColumn}'이 학습 컬럼에 없습니다.")
        return {"status": "error", "message": f"타겟 컬럼 '{targetColumn}'이 학습 컬럼에 없습니다."}

    # original_open = stock_data[targetColumn].values  # ✅ 이 부분 수정 필요
    # dates = pd.to_datetime(stock_data[dateColumn], errors='coerce')  # ✅ dateColumn이 없을 수 있음
    
    # ✅ 날짜 컬럼 처리 (없으면 인덱스 사용)
    if dateColumn in stock_data.columns:
        dates = pd.to_datetime(stock_data[dateColumn], errors='coerce')
    else:
        # 날짜 컬럼이 없으면 가상의 날짜 생성
        dates = pd.date_range(start='2023-01-01', periods=len(stock_data), freq='5T')
        print(f"⚠️ 날짜 컬럼 '{dateColumn}'이 없어서 가상 날짜를 생성했습니다.")
    
    # ✅ 타겟 컬럼 데이터 추출
    original_open = stock_data[targetColumn].values
    
    # ✅ 학습용 데이터 선택
    stock_data_for_training = stock_data[study_columns_list].astype(float)

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

    trainX, trainY, testX, testY = [], [], [], []
    for i in range(seq_len, n_train - pred_days + 1):
        trainX.append(train_data_scaled[i - seq_len:i, 0:input_dim])
        trainY.append(train_data_scaled[i + pred_days - 1:i + pred_days, target_idx])

    for i in range(seq_len, len(test_data_scaled) - pred_days + 1):
        testX.append(test_data_scaled[i - seq_len:i, 0:input_dim])
        testY.append(test_data_scaled[i + pred_days - 1:i + pred_days, target_idx])

    trainX, trainY = np.array(trainX), np.array(trainY)
    testX, testY = np.array(testX), np.array(testY)

    # ✅ Socket.IO 대신 print로 학습 시작 알림
    print(f"================잠시후 {modelName} 모델 학습 시작================")
    print(f"================설정 값: epochs={r_epochs}, batchSize={r_batchSize}, validationSplit={r_validationSplit}, seqLen={r_seqLen}, predDays={r_predDays}================")
    print(f"================날짜 데이터 컬럼: {dateColumn}================")
    print(f"================학습 데이터 컬럼 리스트: {studyColumns}================")
    print(f"================학습 데이터 데이터 수 : {len(lstmData)}================")
    print(f"================예측 데이터 컬럼: {targetColumn}================")

    try:
        model = load_model(model_file_path, compile=False)
        model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
        print("✅ Loaded full model from disk")
    except (OSError, IOError):
        print("🔄 Training model from scratch...")

        model = Sequential([
            Input(shape=(trainX.shape[1], trainX.shape[2])),
            LSTM(64, return_sequences=True),
            LSTM(32, return_sequences=False),
            Dense(trainY.shape[1])
        ])

        model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')

        class TrainingCallback(Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                print(f" Epoch {epoch + 1}: loss={logs.get('loss', 0):.4f}, val_loss={logs.get('val_loss', 0):.4f}")

        history = model.fit(
            trainX, trainY,
            epochs=int(r_epochs),
            batch_size=int(r_batchSize),
            validation_split=float(r_validationSplit),
            verbose=1,  # ✅ verbose=1로 변경하여 진행상황 확인
            callbacks=[TrainingCallback()]
        )

        model.save(model_file_path)
        print("✅ Full model saved successfully.")

        plt.figure(figsize=(14, 5))
        plt.plot(history.history['loss'], label='Training loss')
        plt.plot(history.history['val_loss'], label='Validation loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        plt.savefig(training_loss_path)
        plt.close()

    prediction = model.predict(testX)

    mean_values_pred = np.repeat(scaler.mean_[np.newaxis, :], prediction.shape[0], axis=0)
    mean_values_pred[:, target_idx] = np.squeeze(prediction)
    y_pred = scaler.inverse_transform(mean_values_pred)[:, target_idx]

    mean_values_testY = np.repeat(scaler.mean_[np.newaxis, :], testY.shape[0], axis=0)
    mean_values_testY[:, target_idx] = np.squeeze(testY)
    testY_original = scaler.inverse_transform(mean_values_testY)[:, target_idx]
    valid_test_dates = test_dates[seq_len : seq_len + len(testY_original)]

    plt.figure(figsize=(14, 5))
    plt.plot(dates, original_open, color='green', label='Original '+targetColumn+' Value')
    plt.plot(valid_test_dates, testY_original, color='blue', label='Actual '+targetColumn+' Value')
    plt.plot(valid_test_dates, y_pred, color='red', linestyle='--', label='Predicted '+targetColumn+' Value')
    plt.xlabel(dateColumn); plt.ylabel(targetColumn+' Value')
    plt.title('Original, Actual and Predicted '+targetColumn+' Value')
    plt.legend()
    plt.savefig(total_graph_path)
    plt.close()

    zoom_start = max(0, len(valid_test_dates) - 50)
    plt.figure(figsize=(14, 5))
    plt.plot(valid_test_dates[zoom_start:], testY_original[zoom_start:], color='blue', label='Actual '+targetColumn+' Price')
    plt.plot(valid_test_dates[zoom_start:], y_pred[zoom_start:], color='red', linestyle='--', label='Predicted '+targetColumn+' Price')
    plt.xlabel(dateColumn); plt.ylabel(targetColumn+' Price')
    plt.title('Zoomed In Actual vs Predicted '+targetColumn+' Price')
    plt.legend()
    plt.savefig(diff_graph_path)
    plt.close()

    def mean_absolute_percentage_error(y_true, y_pred):
        mask = y_true != 0
        if np.sum(mask) == 0:
            return 999.0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    mape = mean_absolute_percentage_error(testY_original, y_pred)
    accuracy = 100 - mape if not np.isnan(mape) else np.nan

    print(f"✅ MAPE: {mape:.2f}%")
    print(f"✅ 예측 정확도: {accuracy:.2f}%")

    # config 저장
    config = {
        "targetColumn": targetColumn,
        "dateColumn": dateColumn,
        "studyColumns": studyColumns,
        "r_seqLen": r_seqLen,
        "r_predDays": r_predDays
    }
    with open(os.path.join(model_path, modelName + "_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    # scaler 저장
    joblib.dump(scaler, os.path.join(model_path, modelName + "_scaler.pkl"))

    return {
        "status": "success",
        "training_loss_img": "graphImage/" + modelName + "_trainingLoss.png",
        "total_graph_img": "graphImage/" + modelName + "_totalgraph.png",
        "diff_graph_img": "graphImage/" + modelName + "_diffgraph.png",
        "mape": round(mape, 2),
        "accuracy": round(accuracy, 2)
    }

# ✅ 직접 실행할 때만 작동
if __name__ == "__main__":
    result = lstmLearningStock()
    print("최종 결과:", result)