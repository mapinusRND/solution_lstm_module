# -*- coding: utf-8 -*-
"""
Title   : 외부데이터 기반 LSTM 예측 및 실시간 학습 모듈
Author  : 주성중 / (주)맵인어스
Description: PostgreSQL 데이터를 기반으로 LSTM 모델을 학습하고 시계열 예측을 수행하는 모듈
"""

# ================= 라이브러리 임포트 =================
import os
# TensorFlow 최적화 옵션 비활성화 (경고 메시지 제거)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import absl.logging
# ABSL 로그 제거 (Google 라이브러리 경고)
absl.logging.set_verbosity(absl.logging.ERROR)
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
from sqlalchemy import create_engine  # PostgreSQL 연결을 위한 SQLAlchemy

# ================= 환경 설정 =================
root = "D:/work/lstm"  # 프로젝트 루트 경로

# 모델 저장 경로 설정 및 디렉토리 생성
model_path = os.path.abspath(root + "/saved_models")
os.makedirs(model_path, exist_ok=True)

# ================= 데이터베이스 연결 함수 =================
def get_db_engine():
    """
    SQLAlchemy 엔진을 생성하여 PostgreSQL 연결
    Returns:
        sqlalchemy.engine: 데이터베이스 연결 엔진
    """
    connection_string = "postgresql://[사용자명]:[비밀번호]@[호스트]:[포트]/[데이터베이스명]"
    return create_engine(connection_string)

def get_db_connection():
    """
    psycopg2를 사용한 직접 PostgreSQL 연결
    Returns:
        psycopg2.connection: 데이터베이스 연결 객체
    """
    return psycopg2.connect(
        dbname="디비이름",
        user="사용자명",
        password="비밀번호",
        host="ip",
        port="port"
    )

# ================= LSTM 모델 학습 메인 함수 =================
def lstmLearning():
    """
    LSTM 모델 학습을 시작하는 메인 함수
    데이터베이스에서 데이터를 조회하고 학습 프로세스를 실행
    
    Returns:
        dict: 학습 상태와 결과를 담은 딕셔너리
    """
    # ================= 데이터베이스 조회 설정 =================
    dateColumn = "컬럼명"      # 시계열 날짜/시간 컬럼명
    studyColumns = "컬럼명1,컬럼명2"  # 학습에 사용할 피처 컬럼들 (쉼표로 구분)
    targetColumn = "컬럼명"     # 예측하려는 타겟 컬럼명

    lstmData = None

    # ================= 데이터베이스에서 학습 데이터 조회 =================
    try:    
        tablename = "테이블명"  # 조회할 테이블명
        engine = get_db_engine()     # SQLAlchemy 엔진 생성

        # LSTM 학습용 데이터 조회 쿼리
        query = f"""
        SELECT {studyColumns},{dateColumn}
        FROM carbontwin.{tablename}
        WHERE {dateColumn} IS NOT NULL
        ORDER BY {dateColumn} ASC
        """

        # 쿼리 실행 및 DataFrame으로 변환
        lstmData = pd.read_sql_query(query, engine)
        print(f"✅ 데이터 로드 성공: {len(lstmData)}행")
        
    except Exception as e:
        print(f"❌ 데이터베이스 오류: {str(e)}")
        return {"status": "error", "message": str(e)}
        
    # ================= 백그라운드 학습 작업 실행 =================
    def background_task():
        """
        실제 LSTM 모델 학습을 수행하는 내부 함수
        Returns:
            dict: 학습 결과
        """
        try:
            # 메인 학습 함수 호출
            result = lstmFinance(lstmData, dateColumn, studyColumns, targetColumn)
            print("✅ 백그라운드 작업 성공:", result)
            return result
        except Exception as e:
            print("❌ 백그라운드 작업 오류:", str(e))
            return {"status": "error", "message": str(e)}

    # 백그라운드 작업 실행
    result = background_task()
    
    return {"status": "success", "message": "학습이 시작되었습니다.", "result": result}

# ================= LSTM 모델 학습 및 예측 함수 =================
def lstmFinance(lstmData, dateColumn, studyColumns, targetColumn):
    """
    LSTM 모델의 학습, 예측, 평가를 수행하는 메인 함수
    
    Args:
        lstmData (DataFrame): 학습용 데이터
        dateColumn (str): 날짜/시간 컬럼명
        studyColumns (str): 학습 피처 컬럼들 (쉼표로 구분)
        targetColumn (str): 예측 타겟 컬럼명
        
    Returns:
        dict: 학습 및 예측 결과
    """
    # TensorFlow eager execution 활성화 (디버깅 용이)
    if not tf.executing_eagerly():
        tf.config.run_functions_eagerly(True)

    # ================= 하이퍼파라미터 입력 받기 =================
    try:
        modelName = input("모델명: ").strip() or "training_1"
        r_epochs = int(input("에포크 수 (기본값: 20): ").strip() or "20")
        r_batchSize = int(input("배치 크기 (기본값: 16): ").strip() or "16")
        r_validationSplit = float(input("검증 데이터 비율 (기본값: 0.1): ").strip() or "0.1")
        r_seqLen = int(input("시퀀스 길이 (기본값: 14): ").strip() or "14")
        r_predDays = int(input("예측 일수 (기본값: 1): ").strip() or "1")
    except ValueError:
        print("❌ 잘못된 입력값입니다. 기본값을 사용합니다.")
        # 기본 하이퍼파라미터 설정
        modelName = "training_1"
        r_epochs = 20           # 학습 에포크 수
        r_batchSize = 16        # 배치 크기
        r_validationSplit = 0.1 # 검증 데이터 비율 (10%)
        r_seqLen = 14          # 입력 시퀀스 길이 (14일간의 데이터)
        r_predDays = 1         # 예측할 미래 일수

    # 모델 파일 저장 경로 설정
    model_file_path = os.path.join(model_path, modelName + ".h5")

    # ================= 데이터 검증 =================
    if lstmData.empty:
        print(f"❌ 데이터가 비어있습니다.")
        return {"status": "error", "message": "데이터가 비어있습니다."}
    
    # 학습 컬럼 리스트 생성
    study_columns_list = [col.strip() for col in studyColumns.split(',')]
    
    # 타겟 컬럼이 학습 컬럼에 포함되어 있는지 확인
    if targetColumn not in study_columns_list:
        print(f"❌ 타겟 컬럼 '{targetColumn}'이 학습 컬럼에 없습니다.")
        return {"status": "error", "message": f"타겟 컬럼 '{targetColumn}'이 학습 컬럼에 없습니다."}
    
    # ================= 날짜 데이터 처리 =================
    if dateColumn in lstmData.columns:
        # 날짜 컬럼이 존재하면 datetime으로 변환
        dates = pd.to_datetime(lstmData[dateColumn], errors='coerce')
    else:
        # 날짜 컬럼이 없으면 가상의 날짜 범위 생성 (5분 간격)
        dates = pd.date_range(start='2023-01-01', periods=len(lstmData), freq='5T')
        print(f"⚠️ 날짜 컬럼 '{dateColumn}'이 없어서 가상 날짜를 생성했습니다.")
    
    # ================= 데이터 전처리 =================
    # 학습용 피처 데이터 선택 및 float 타입으로 변환
    stock_data_for_training = lstmData[study_columns_list].astype(float)

    # 데이터 정규화 (StandardScaler 사용)
    scaler = StandardScaler()
    stock_data_scaled = scaler.fit_transform(stock_data_for_training)

    # ================= 훈련/테스트 데이터 분할 =================
    n_train = int(0.9 * stock_data_scaled.shape[0])  # 90%를 훈련 데이터로 사용
    train_data_scaled = stock_data_scaled[:n_train]   # 훈련 데이터
    test_data_scaled = stock_data_scaled[n_train:]    # 테스트 데이터
    test_dates = dates[n_train:]                      # 테스트 데이터의 날짜

    # ================= LSTM 입력 파라미터 설정 =================
    pred_days = int(r_predDays)                              # 예측할 미래 일수
    seq_len = int(r_seqLen)                                 # 입력 시퀀스 길이
    input_dim = stock_data_for_training.shape[1]            # 입력 피처 수
    target_idx = study_columns_list.index(targetColumn)     # 타겟 컬럼의 인덱스

    # ================= 시계열 데이터셋 생성 =================
    trainX, trainY, testX, testY = [], [], [], []
    
    # 훈련 데이터셋 생성
    # seq_len만큼의 과거 데이터로 pred_days 후의 값을 예측
    for i in range(seq_len, n_train - pred_days + 1):
        trainX.append(train_data_scaled[i - seq_len:i, 0:input_dim])  # 입력: seq_len개의 시점 데이터
        trainY.append(train_data_scaled[i + pred_days - 1:i + pred_days, target_idx])  # 출력: pred_days 후 타겟 값

    # 테스트 데이터셋 생성
    for i in range(seq_len, len(test_data_scaled) - pred_days + 1):
        testX.append(test_data_scaled[i - seq_len:i, 0:input_dim])
        testY.append(test_data_scaled[i + pred_days - 1:i + pred_days, target_idx])

    # 리스트를 numpy 배열로 변환
    trainX, trainY = np.array(trainX), np.array(trainY)
    testX, testY = np.array(testX), np.array(testY)

    # ================= 학습 시작 알림 =================
    print(f"================잠시후 {modelName} 모델 학습 시작================")
    print(f"================설정 값: epochs={r_epochs}, batchSize={r_batchSize}, validationSplit={r_validationSplit}, seqLen={r_seqLen}, predDays={r_predDays}================")
    print(f"================날짜 데이터 컬럼: {dateColumn}================")
    print(f"================학습 데이터 컬럼 리스트: {studyColumns}================")
    print(f"================학습 데이터 데이터 수 : {len(lstmData)}================")
    print(f"================예측 데이터 컬럼: {targetColumn}================")

    # ================= 모델 로드 또는 생성 =================
    try:
        # 기존 모델이 있으면 로드
        model = load_model(model_file_path, compile=False)
        model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
        print("✅ Loaded full model from disk")
    except (OSError, IOError):
        # 모델이 없으면 새로 생성
        print("🔄 Training model from scratch...")

        # LSTM 모델 구조 정의
        model = Sequential([
            Input(shape=(trainX.shape[1], trainX.shape[2])),  # 입력 레이어
            LSTM(64, return_sequences=True),                   # 첫 번째 LSTM 레이어 (64 유닛)
            LSTM(32, return_sequences=False),                  # 두 번째 LSTM 레이어 (32 유닛)
            Dense(trainY.shape[1])                            # 출력 레이어 (Dense)
        ])

        # 모델 컴파일 (옵티마이저: Adam, 손실함수: MSE)
        model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')

        # ================= 커스텀 콜백 클래스 정의 =================
        class TrainingCallback(Callback):
            """
            에포크마다 학습 진행상황을 출력하는 커스텀 콜백
            """
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                print(f" Epoch {epoch + 1}: loss={logs.get('loss', 0):.4f}, val_loss={logs.get('val_loss', 0):.4f}")

        # ================= 모델 학습 실행 =================
        history = model.fit(
            trainX, trainY,                           # 훈련 데이터
            epochs=int(r_epochs),                     # 에포크 수
            batch_size=int(r_batchSize),             # 배치 크기
            validation_split=float(r_validationSplit), # 검증 데이터 비율
            verbose=1,                                # 학습 진행상황 출력
            callbacks=[TrainingCallback()]            # 커스텀 콜백 적용
        )

        # 학습된 모델 저장
        model.save(model_file_path)
        print("✅ Full model saved successfully.")

    # ================= 모델 예측 수행 =================
    prediction = model.predict(testX)

    # ================= 예측 결과 역정규화 =================
    # 예측값을 원래 스케일로 복원
    mean_values_pred = np.repeat(scaler.mean_[np.newaxis, :], prediction.shape[0], axis=0)
    mean_values_pred[:, target_idx] = np.squeeze(prediction)
    y_pred = scaler.inverse_transform(mean_values_pred)[:, target_idx]

    # 실제값을 원래 스케일로 복원
    mean_values_testY = np.repeat(scaler.mean_[np.newaxis, :], testY.shape[0], axis=0)
    mean_values_testY[:, target_idx] = np.squeeze(testY)
    testY_original = scaler.inverse_transform(mean_values_testY)[:, target_idx]
    
    # 테스트 데이터에 해당하는 날짜 추출
    valid_test_dates = test_dates[seq_len : seq_len + len(testY_original)]

    # ================= 모델 성능 평가 =================
    def mean_absolute_percentage_error(y_true, y_pred):
        """
        평균 절대 백분율 오차(MAPE) 계산
        Args:
            y_true: 실제값
            y_pred: 예측값
        Returns:
            float: MAPE 값 (%)
        """
        mask = y_true != 0  # 0으로 나누는 것을 방지
        if np.sum(mask) == 0:
            return 999.0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    # MAPE와 정확도 계산
    mape = mean_absolute_percentage_error(testY_original, y_pred)
    accuracy = 100 - mape if not np.isnan(mape) else np.nan

    # ================= 결과 출력 =================
    print(f"✅ MAPE: {mape:.2f}%")
    print(f"✅ 예측 정확도: {accuracy:.2f}%")

# ================= 메인 실행 부분 =================
if __name__ == "__main__":
    """
    스크립트가 직접 실행될 때만 동작
    """
    result = lstmLearning()
    print("최종 결과:", result)