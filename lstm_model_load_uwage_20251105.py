# -*- coding: utf-8 -*-
"""
Title   : LSTM 예측 - 개선된 역정규화 방식
Author  : 주성중 / (주)맵인어스
Description: 
    - ✅ 스케일러 독립적인 역정규화 방식
    - ✅ 전체 피처 벡터를 활용한 안전한 변환
    - ✅ StandardScaler, MinMaxScaler 등 모두 지원
Version : 9.0 (역정규화 개선)
Date    : 2025-10-28
"""

# ============================================================================
# 환경 설정 및 라이브러리 임포트
# ============================================================================

import os
# TensorFlow 최적화 옵션 비활성화 (경고 메시지 억제)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# TensorFlow 로그 레벨 설정 (ERROR만 출력)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
# 경고 메시지 무시 설정
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import joblib
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta

# ============================================================================
# 환경별 경로 설정
# ============================================================================

# Flask 환경 변수를 통해 로컬/서버 환경 구분
root = "D:/work/lstm"

# 모델 저장 경로 및 예측 결과 저장 경로 설정
model_path = os.path.abspath(root + "/saved_models")
prediction_path = os.path.abspath(root + "/predictions")
os.makedirs(prediction_path, exist_ok=True)

# ============================================================================
# 데이터베이스 연결 함수
# ============================================================================

def get_db_engine():
    """
    PostgreSQL 데이터베이스 연결 엔진 생성
    
    Returns:
        SQLAlchemy Engine 객체
    """
    # connection_string = "postgresql://postgres:mapinus@10.10.10.201:5432/postgres"
    connection_string = "postgresql://postgres:mapinus%401004@10.10.10.201:5434/postgres"
    # connection_string = "postgresql://postgres:carbontwin@221.150.43.89:15432/postgres"
    return create_engine(connection_string)

# ============================================================================
# 데이터 직렬화 함수
# ============================================================================

def convert_to_serializable(obj):
    """
    NumPy, Pandas 객체를 JSON 직렬화 가능한 Python 기본 타입으로 변환
    
    Args:
        obj: 변환할 객체 (ndarray, int64, float64, Timestamp 등)
    
    Returns:
        직렬화 가능한 Python 기본 타입 (list, int, float, str)
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    return obj

# ============================================================================
# 데이터베이스에서 데이터 로드
# ============================================================================

def load_new_data(tablename, dateColumn, studyColumns):
    """
    데이터베이스에서 학습/예측에 필요한 데이터 로드
    
    Args:
        tablename (str): 테이블명 (예: lstm_input_15m_new)
        dateColumn (str): 날짜 컬럼명 (예: time_point)
        studyColumns (str): 분석할 컬럼들 (쉼표로 구분, 예: 'usage_kwh,week_code,is_weekend')
    
    Returns:
        pandas.DataFrame: 시계열 순으로 정렬된 데이터
        None: 로드 실패 시
    """
    try:
        engine = get_db_engine()
        
        # SQL 쿼리 작성 (날짜 기준 오름차순 정렬)
        query = f"""
            SELECT {studyColumns},{dateColumn}
            FROM carbontwin.{tablename}
            WHERE {dateColumn} IS NOT NULL
              AND time_point >= (
                    SELECT MAX(time_point) - INTERVAL '1 days'
                    FROM carbontwin.{tablename}
                    WHERE time_point IS NOT null
                )
            ORDER BY {dateColumn} ASC
            """
        
        # 데이터 로드
        data = pd.read_sql_query(query, engine)
        print(f"✅ 데이터 로드: {len(data)}행")
        
        # 날짜 컬럼을 datetime 타입으로 변환 및 기간 출력
        if len(data) > 0 and dateColumn in data.columns:
            data[dateColumn] = pd.to_datetime(data[dateColumn])
            min_date = data[dateColumn].min()
            max_date = data[dateColumn].max()
            print(f"   📅 기간: {min_date} ~ {max_date}")
        
        return data
        
    except Exception as e:
        print(f"❌ 데이터 로드 오류: {str(e)}")
        return None

# ============================================================================
# 학습된 LSTM 모델 로드
# ============================================================================

def load_trained_model(model_name):
    """
    저장된 LSTM 모델, 스케일러, 설정 파일 로드
    
    Args:
        model_name (str): 모델명 (예: usage-kwh-model-4)
    
    Returns:
        tuple: (model, scaler, config)
            - model: Keras LSTM 모델
            - scaler: sklearn 스케일러 (StandardScaler/MinMaxScaler)
            - config: 모델 학습 시 사용된 설정 정보 (dict)
        (None, None, None): 로드 실패 시
    """
    try:
        # 필요한 파일 경로 설정
        model_file = os.path.join(model_path, f"{model_name}.h5")
        scaler_file = os.path.join(model_path, f"{model_name}_scaler.pkl")
        config_file = os.path.join(model_path, f"{model_name}_config.json")
        
        # 파일 존재 여부 확인
        if not all(os.path.exists(f) for f in [model_file, scaler_file, config_file]):
            print(f"❌ 필요한 파일을 찾을 수 없습니다.")
            return None, None, None
        
        print(f"📂 모델 로드: {model_name}")
        
        # 경고 메시지 억제하면서 모델 및 스케일러 로드
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = load_model(model_file, compile=False)
            model.compile(optimizer='adam', loss='mse')
            scaler = joblib.load(scaler_file)
        
        # 설정 파일 로드 (JSON)
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 로드된 정보 출력
        print(f"✅ 모델 로드 완료")
        print(f"   - 타겟: {config['targetColumn']}")
        print(f"   - 시퀀스: {config['r_seqLen']}")
        print(f"   - 스케일러: {type(scaler).__name__}")
        
        return model, scaler, config
        
    except Exception as e:
        print(f"❌ 모델 로드 오류: {str(e)}")
        return None, None, None

# ============================================================================
# 실제 데이터로 모델 검증
# ============================================================================

def validate_with_actual_data(model, scaler, config, data, validation_days=1):
    """
    최근 N일의 실제 데이터로 모델 성능 검증 (개선된 역정규화 방식)
    
    Args:
        model: LSTM 모델
        scaler: 데이터 스케일러
        config (dict): 모델 설정 정보
        data (DataFrame): 전체 데이터
        validation_days (int): 검증할 최근 일수 (기본 7일)
    
    Returns:
        dict: 검증 결과
            - status: 성공/실패 상태
            - statistics: 정확도, MAPE, MAE, RMSE
            - historical_mean: 검증 데이터의 평균값
            - historical_std: 검증 데이터의 표준편차
        None: 검증 실패 시
    
    검증 프로세스:
        1. 최근 N일 데이터를 검증 세트로 분리
        2. 시퀀스 데이터 생성 (seq_len 길이의 입력 → r_predDays 후의 출력)
        3. 모델 예측 수행
        4. 개선된 역정규화 방식 적용 (전체 피처 벡터 활용)
        5. 성능 지표 계산 (정확도, MAPE, MAE, RMSE)
    """
    try:
        print(f"\n{'='*80}")
        print(f"🔍 모델 검증 시작 (최근 {validation_days}일)")
        print(f"{'='*80}")
        
        # 설정 정보 추출
        dateColumn = config['dateColumn']
        studyColumns = config['studyColumns']
        targetColumn = config['targetColumn']
        seq_len = int(config['r_seqLen'])  # 입력 시퀀스 길이 (예: 96 = 1일치 15분 데이터)
        r_predDays = int(config.get('r_predDays', 1))  # 예측할 미래 스텝 수
        
        # 컬럼 리스트 생성 및 타겟 컬럼 인덱스 찾기
        study_columns_list = [col.strip() for col in studyColumns.split(',')]
        target_idx = study_columns_list.index(targetColumn)
        
        # 예측용 데이터 준비 (숫자형으로 변환)
        data_for_prediction = data[study_columns_list].astype(float)
        dates = pd.to_datetime(data[dateColumn])
        
        # 검증 데이터 범위 계산 (96개/일 * N일)
        validation_points = 96 * validation_days
        validation_start_idx = len(data) - validation_points - r_predDays
        
        # 데이터 정규화 (학습 시 사용한 스케일러 적용)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data_scaled = scaler.transform(data_for_prediction)
        
        # 시퀀스 데이터 생성 (검증용)
        testX, testY = [], []
        test_range = range(seq_len, len(data_scaled) - r_predDays + 1)
        
        for i in test_range:
            # 검증 시작 지점 이전 데이터는 제외
            if i < validation_start_idx:
                continue
            # X: 과거 seq_len 스텝의 모든 피처
            testX.append(data_scaled[i - seq_len:i, :].astype(np.float32))
            # Y: r_predDays 후의 타겟 값
            testY.append(data_scaled[i + r_predDays - 1:i + r_predDays, target_idx].astype(np.float32))
        
        testX = np.array(testX, dtype=np.float32)
        testY = np.array(testY, dtype=np.float32)
        
        print(f"\n🔄 역정규화 방식: 전체 피처 벡터 활용 (스케일러 독립적)")
        
        # 모델 예측 및 역정규화
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 1. 모델 예측 (정규화된 값)
            prediction = model.predict(testX, verbose=0)
            
            # ✅ 개선된 역정규화: 전체 피처 벡터 방식
            # - StandardScaler, MinMaxScaler 등 모든 스케일러에 대해 안전하게 작동
            # - 각 피처 간의 상관관계를 유지하면서 역변환
            
            # 예측값 역정규화
            y_pred = []
            for i, pred_scaled in enumerate(prediction):
                # testX[i]의 마지막 타임스텝을 베이스로 사용
                full_scaled = testX[i, -1, :].copy()  # 마지막 타임스텝의 전체 피처
                full_scaled[target_idx] = pred_scaled[0]  # 타겟 위치에 예측값 삽입
                # 전체 피처를 한 번에 역변환
                full_original = scaler.inverse_transform(full_scaled.reshape(1, -1))[0]
                y_pred.append(full_original[target_idx])
            y_pred = np.array(y_pred)
            
            # 실제값 역정규화
            testY_original = []
            for i, y_scaled in enumerate(testY):
                full_scaled = testX[i, -1, :].copy()  # 마지막 타임스텝의 전체 피처
                full_scaled[target_idx] = y_scaled[0]  # 타겟 위치에 실제값 삽입
                # 전체 피처를 한 번에 역변환
                full_original = scaler.inverse_transform(full_scaled.reshape(1, -1))[0]
                testY_original.append(full_original[target_idx])
            testY_original = np.array(testY_original)
        
        # 성능 지표 계산
        eps = 9  
        mask = testY_original > eps  # 임계값보다 큰 값만 사용
        # MAPE: Mean Absolute Percentage Error (평균 절대 백분율 오차)
        mape = np.mean(np.abs((y_pred[mask] - testY_original[mask]) / testY_original[mask])) * 100 if np.sum(mask) > 0 else 999.0
        
        # 정확도 = 100 - MAPE
        accuracy = 100 - mape
        # MAE: Mean Absolute Error (평균 절대 오차)
        mae = np.mean(np.abs(y_pred - testY_original))
        # RMSE: Root Mean Square Error (평균 제곱근 오차)
        rmse = np.sqrt(np.mean((y_pred - testY_original) ** 2))
        
        # 검증 결과 출력
        print(f"\n📊 검증 결과:")
        print(f"   정확도: {accuracy:.2f}%")
        print(f"   MAPE:   {mape:.2f}%")
        print(f"   MAE:    {mae:.4f}")
        print(f"   RMSE:   {rmse:.4f}")
        
        return {
            "status": "success",
            "statistics": {"accuracy": accuracy, "mape": mape, "mae": mae, "rmse": rmse},
            "historical_mean": np.mean(testY_original),
            "historical_std": np.std(testY_original)
        }
        
    except Exception as e:
        print(f"❌ 검증 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# 미래 전력 사용량 예측 (안정화 최소화 버전)
# ============================================================================

def predict_future_stable(model, scaler, config, data, future_steps=672, historical_mean=None, historical_std=None):
    """
    LSTM 모델을 사용한 미래 전력 사용량 예측 (안정화 최소화)
    
    Args:
        model: LSTM 모델
        scaler: 데이터 스케일러
        config (dict): 모델 설정
        data (DataFrame): 전체 과거 데이터
        future_steps (int): 예측할 미래 스텝 수 (기본 672 = 7일 * 96)
        historical_mean (float): 과거 데이터 평균 (검증에서 전달)
        historical_std (float): 과거 데이터 표준편차 (검증에서 전달)
    
    Returns:
        dict: 예측 결과
            - metadata: 모델 정보, 예측 방법, 학습된 패턴 등
            - predictions: 각 시점별 예측값 리스트
            - statistics: 예측값의 통계 (최소, 최대, 평균, 표준편차)
        None: 예측 실패 시
    
    예측 프로세스:
        1. 평일/휴일 패턴 학습 (과거 데이터 분석)
        2. 마지막 seq_len 길이의 데이터로 초기 시퀀스 구성
        3. 반복적으로 다음 스텝 예측:
           - 모델로 예측 (정규화된 값)
           - 역정규화 (전체 피처 벡터 방식)
           - 극단적 이상치만 제거 (5σ 범위)
           - 시간 특성 업데이트 (요일, 주말 여부 등)
           - 예측값을 시퀀스에 추가하여 다음 예측 준비
        4. 결과 통계 및 패턴 비교
    """
    
    # ------------------------------------------------------------------------
    # 내부 함수 1: 평일/휴일 패턴 학습
    # ------------------------------------------------------------------------
    def calculate_workday_holiday_patterns(data_for_prediction, dates, targetColumn):
        """
        과거 데이터에서 평일/휴일 패턴을 자동으로 학습
        (데이터 부족 시 안전하게 처리)
        """
        print(f"   🔍 평일/휴일 패턴 학습 중...")
        
        target_values = data_for_prediction[targetColumn].values
        # 평일 마스크 (월~금: 0~4)
        weekday_mask = dates.dt.weekday < 5
        # 주말 마스크 (토~일: 5~6)
        weekend_mask = dates.dt.weekday >= 5
        
        weekday_values = target_values[weekday_mask]
        weekend_values = target_values[weekend_mask]
        
        # 🔥 평일/휴일 통계 정보 계산 (데이터 없을 경우 대비)
        def safe_stats(values, name):
            """빈 배열에도 안전한 통계 계산"""
            if len(values) == 0:
                print(f"      ⚠️  {name} 데이터 없음 → 기본값 사용")
                return {
                    "mean": 0.0,
                    "std": 0.0,
                    "median": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "q25": 0.0,
                    "q75": 0.0,
                    "zero_ratio": 1.0,
                    "count": 0
                }
            
            return {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "q25": float(np.percentile(values, 25)),
                "q75": float(np.percentile(values, 75)),
                "zero_ratio": float(np.sum(values == 0) / len(values)),
                "count": len(values)
            }
        
        patterns = {
            "workday": safe_stats(weekday_values, "평일"),
            "holiday": safe_stats(weekend_values, "휴일")
        }
        
        # 요일별 상세 정보 계산
        weekday_names = ['월', '화', '수', '목', '금', '토', '일']
        weekday_details = {}
        
        for day_idx in range(7):
            day_mask = dates.dt.weekday == day_idx
            day_values = target_values[day_mask]
            
            if len(day_values) > 0:
                weekday_details[day_idx] = {
                    "name": weekday_names[day_idx],
                    "mean": float(np.mean(day_values)),
                    "std": float(np.std(day_values)),
                    "zero_ratio": float(np.sum(day_values == 0) / len(day_values)),
                    "count": len(day_values),
                    "is_workday": day_idx < 5
                }
        
        return patterns, weekday_details
    
    # ------------------------------------------------------------------------
    # 내부 함수 2: 예측값 안정화 (극단적 이상치만 제거)
    # ------------------------------------------------------------------------
    def adaptive_stabilization(pred_original, next_date, patterns):
        """
        ✅ 예측값을 거의 그대로 사용 (극단적 이상치만 제거)
        (데이터 부족 시 안전하게 처리)
        """
        day_of_week = next_date.weekday()
        is_workday = day_of_week < 5
        
        # 평일/휴일 패턴 선택
        if is_workday:
            pattern = patterns["workday"]
            day_type = "평일"
            icon = "🏢"
        else:
            pattern = patterns["holiday"]
            day_type = "휴일"
            icon = "🏖️"
        
        weekday_names = ['월', '화', '수', '목', '금', '토', '일']
        weekday_name = weekday_names[day_of_week]
        
        mean = pattern["mean"]
        std = pattern["std"]
        
        stabilization_applied = False
        stabilization_reason = ""
        
        # 🔥 데이터가 없는 경우 (count == 0) 또는 표준편차가 0인 경우
        if pattern["count"] == 0 or std == 0:
            # 안정화 없이 그대로 사용 (단, 음수만 제거)
            pred_original = max(0, pred_original)
            return pred_original, False, "데이터 부족(안정화 스킵)", weekday_name, day_type, icon
        
        # 정상적인 안정화 (5σ 범위)
        safe_min = max(0, mean - 5 * std)
        safe_max = mean + 5 * std
        
        # 극단적 이상치만 제거
        if pred_original < safe_min:
            pred_original = safe_min
            stabilization_applied = True
            stabilization_reason = f"극단적 최소값 ({safe_min:.1f} 미만)"
        elif pred_original > safe_max:
            pred_original = safe_max
            stabilization_applied = True
            stabilization_reason = f"극단적 최대값 ({safe_max:.1f} 초과)"
        
        # 음수 방지
        pred_original = max(0, pred_original)
        
        return pred_original, stabilization_applied, stabilization_reason, weekday_name, day_type, icon
    
    # ------------------------------------------------------------------------
    # 내부 함수 3: 시간 특성 업데이트
    # ------------------------------------------------------------------------
    def update_time_features(next_row, next_date, study_columns_list):
        """
        시간 관련 특성 동적 업데이트
        
        Args:
            next_row (array): 다음 시점의 피처 벡터
            next_date (datetime): 다음 시점의 날짜
            study_columns_list (list): 피처 컬럼 리스트
        
        Returns:
            array: 시간 특성이 업데이트된 피처 벡터
        
        업데이트 항목:
        - week_code: 요일 코드 (1~7)
        - is_weekend: 주말 여부 (0 or 1)
        - is_workday: 평일 여부 (0 or 1)
        - day_sin, day_cos: 요일의 순환 특성 (사인/코사인 인코딩)
        
        이유: 시계열 예측 시 요일 정보가 중요한 피처이므로
              다음 시점의 요일 정보를 정확히 반영해야 함
        """
        day_of_week = next_date.weekday()
        
        # week_code: 요일 코드 업데이트 (1=월 ~ 7=일)
        # if 'week_code' in study_columns_list:
        #     idx = study_columns_list.index('week_code')
        #     next_row[idx] = min(day_of_week + 1, 6)
        
        # is_weekend: 주말 여부 (토, 일 = 1)
        if 'is_weekend' in study_columns_list:
            idx = study_columns_list.index('is_weekend')
            next_row[idx] = 1 if day_of_week >= 5 else 0
        
        # is_workday: 평일 여부 (월~금 = 1)
        if 'is_workday' in study_columns_list:
            idx = study_columns_list.index('is_workday')
            next_row[idx] = 1 if day_of_week < 5 else 0
        
        # day_sin, day_cos: 요일의 순환적 특성 인코딩
        # (7일 주기를 원형으로 표현하여 월요일과 일요일의 연속성 반영)
        if 'day_sin' in study_columns_list:
            idx = study_columns_list.index('day_sin')
            next_row[idx] = np.sin(2 * np.pi * day_of_week / 7)
        
        if 'day_cos' in study_columns_list:
            idx = study_columns_list.index('day_cos')
            next_row[idx] = np.cos(2 * np.pi * day_of_week / 7)
        
        return next_row
    
    # ========================================================================
    # 메인 예측 로직 시작
    # ========================================================================
    try:
        print(f"\n{'='*80}")
        print(f"🔮 평일/휴일 기반 예측 ({future_steps}개 스텝 = {future_steps//96}일)")
        print(f"{'='*80}")
        
        # 설정 정보 추출
        dateColumn = config['dateColumn']
        studyColumns = config['studyColumns']
        targetColumn = config['targetColumn']
        seq_len = int(config['r_seqLen'])
        r_predDays = int(config.get('r_predDays', 1))
        
        # 컬럼 정보 준비
        study_columns_list = [col.strip() for col in studyColumns.split(',')]
        target_idx = study_columns_list.index(targetColumn)
        
        # 데이터 준비
        data_for_prediction = data[study_columns_list].astype(float)
        dates = pd.to_datetime(data[dateColumn])
        last_date = dates.iloc[-1]  # 마지막 알려진 날짜
        
        # 평일/휴일 패턴 학습
        patterns, weekday_details = calculate_workday_holiday_patterns(
            data_for_prediction, dates, targetColumn
        )
        
        # 과거 데이터 통계 (검증에서 전달되지 않은 경우 계산)
        if historical_mean is None:
            historical_mean = data_for_prediction[targetColumn].mean()
        if historical_std is None:
            historical_std = data_for_prediction[targetColumn].std()
        
        # 학습된 패턴 출력
        # 학습된 패턴 출력 (메인 예측 로직 내)
        print(f"\n   📊 학습된 패턴:")
        print(f"      🏢 평일 (월~금):")
        if patterns['workday']['count'] > 0:
            print(f"         - 평균: {patterns['workday']['mean']:6.2f} kWh (±{patterns['workday']['std']:5.2f})")
            print(f"         - 범위: [{patterns['workday']['min']:.2f}, {patterns['workday']['max']:.2f}]")
            print(f"         - 0값 비율: {patterns['workday']['zero_ratio']*100:4.1f}%")
            print(f"         - 데이터 수: {patterns['workday']['count']:,}개")
        else:
            print(f"         ⚠️  데이터 없음 (기본값 사용)")

        print(f"\n      🏖️ 휴일 (토, 일):")
        if patterns['holiday']['count'] > 0:
            print(f"         - 평균: {patterns['holiday']['mean']:6.2f} kWh (±{patterns['holiday']['std']:5.2f})")
            print(f"         - 범위: [{patterns['holiday']['min']:.2f}, {patterns['holiday']['max']:.2f}]")
            print(f"         - 0값 비율: {patterns['holiday']['zero_ratio']*100:4.1f}%")
            print(f"         - 데이터 수: {patterns['holiday']['count']:,}개")
        else:
            print(f"         ⚠️  데이터 없음 (기본값 사용)")

        print(f"\n   📅 요일별 상세:")
        for day_idx in range(7):
            if day_idx in weekday_details:
                detail = weekday_details[day_idx]
                icon = "🏢" if detail["is_workday"] else "🏖️"
                print(f"      {icon} {detail['name']}요일: {detail['mean']:6.2f} kWh "
                    f"(±{detail['std']:5.2f}) | 0값: {detail['zero_ratio']*100:4.1f}%")
        
        print(f"\n   🔄 역정규화: 전체 피처 벡터 방식")
        print(f"   ✅ 안정화: 5σ 범위 (극단적 이상치만 제거)")
        
        # 데이터 정규화
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data_scaled = scaler.transform(data_for_prediction)
        
        # 초기 시퀀스 구성 (마지막 seq_len 길이의 데이터)
        current_sequence = data_scaled[-seq_len:, :].copy()
        
        # 예측 결과 저장용 리스트
        future_predictions = []
        future_dates = []
        stabilization_log = []  # 안정화 적용 이력
        max_log = 10  # 로그 최대 개수
        
        # ====================================================================
        # 반복적 예측 루프 (future_steps 만큼 반복)
        # ====================================================================
        for step in range(future_steps):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # 1) 모델 예측 (정규화된 값)
                X = current_sequence.reshape(1, seq_len, -1)
                pred_scaled = model.predict(X, verbose=0)[0, 0]
                
                # 2) 역정규화 (전체 피처 벡터 방식)
                full_scaled = current_sequence[-1].copy()
                full_scaled[target_idx] = pred_scaled
                full_original = scaler.inverse_transform(full_scaled.reshape(1, -1))[0]
                pred_original = float(full_original[target_idx])
            
            # 3) 다음 시점의 날짜 계산
            next_date = last_date + timedelta(minutes=15 * (step + 1))
            
            # 4) 안정화 적용 (극단적 이상치만 제거)
            pred_original, stabilized, reason, weekday_name, day_type, icon = adaptive_stabilization(
                pred_original, next_date, patterns
            )
            
            # 5) 안정화 로그 기록 (처음 10건만)
            if stabilized and len(stabilization_log) < max_log:
                stabilization_log.append({
                    "step": step,
                    "date": next_date.strftime("%m-%d %H:%M"),
                    "weekday": weekday_name,
                    "type": day_type,
                    "icon": icon,
                    "value": pred_original,
                    "reason": reason
                })
            
            # 6) 예측 결과 저장
            future_predictions.append(pred_original)
            future_dates.append(next_date)
            
            # 7) 다음 예측을 위한 시퀀스 업데이트
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # 예측값을 포함한 다음 행 생성
                next_row = data_for_prediction.iloc[-1].copy().values
                next_row[target_idx] = pred_original
                # 시간 특성 업데이트 (요일 등)
                next_row = update_time_features(next_row, next_date, study_columns_list)
                # 정규화
                next_row_scaled = scaler.transform(next_row.reshape(1, -1))[0].astype(np.float32)
                
                # NaN/Inf 체크 및 안전 처리
                if np.any(np.isnan(next_row_scaled)) or np.any(np.isinf(next_row_scaled)):
                    next_row_scaled = np.mean(current_sequence[-10:], axis=0)
            
            # 시퀀스 슬라이딩 (맨 앞 제거, 맨 뒤 추가)
            current_sequence = np.vstack([current_sequence[1:], next_row_scaled.reshape(1, -1)])
            
            # 진행 상황 출력 (1일 단위)
            if (step + 1) % 96 == 0:
                print(f"   ⏳ {step + 1}/{future_steps} 완료 ({(step+1)//96}일)")
        
        # 안정화 로그 출력
        if stabilization_log:
            print(f"\n   ⚠️  안정화 적용 사례 (총 {len(stabilization_log)}건):")
            for log in stabilization_log[:5]:
                print(f"      {log['icon']} {log['date']} ({log['weekday']}, {log['type']}): "
                      f"{log['value']:.2f} kWh - {log['reason']}")
            if len(stabilization_log) > 5:
                print(f"      ... 외 {len(stabilization_log) - 5}건")
        
        # NumPy 배열로 변환
        future_predictions = np.array(future_predictions)
        
        # ====================================================================
        # 예측 결과 통계 출력
        # ====================================================================
        print(f"\n📊 예측 결과:")
        print(f"   - 최소: {np.min(future_predictions):.2f} kWh")
        print(f"   - 최대: {np.max(future_predictions):.2f} kWh")
        print(f"   - 평균: {np.mean(future_predictions):.2f} kWh")
        print(f"   - 표준편차: {np.std(future_predictions):.2f} kWh")
        
        # 평일/휴일별 예측 분리
        workday_predictions = []
        holiday_predictions = []
        
        for pred_val, pred_date in zip(future_predictions, future_dates):
            if pred_date.weekday() < 5:
                workday_predictions.append(pred_val)
            else:
                holiday_predictions.append(pred_val)
        
        # 학습 데이터와 예측 데이터 비교
        print(f"\n   📅 예측된 평일/휴일 평균 (vs 학습 데이터):")
        
        if workday_predictions:
            pred_workday_avg = np.mean(workday_predictions)
            actual_workday_avg = patterns["workday"]["mean"]
            diff = pred_workday_avg - actual_workday_avg
            diff_pct = (diff / actual_workday_avg * 100) if actual_workday_avg > 0 else 0
            print(f"      🏢 평일: {pred_workday_avg:6.2f} kWh "
                  f"(학습: {actual_workday_avg:6.2f}, 차이: {diff:+6.2f} / {diff_pct:+5.1f}%)")
        
        if holiday_predictions:
            pred_holiday_avg = np.mean(holiday_predictions)
            actual_holiday_avg = patterns["holiday"]["mean"]
            diff = pred_holiday_avg - actual_holiday_avg
            diff_pct = (diff / actual_holiday_avg * 100) if actual_holiday_avg > 0 else 0
            print(f"      🏖️ 휴일: {pred_holiday_avg:6.2f} kWh "
                  f"(학습: {actual_holiday_avg:6.2f}, 차이: {diff:+6.2f} / {diff_pct:+5.1f}%)")
        
        # ====================================================================
        # 결과를 JSON 직렬화 가능한 형태로 변환
        # ====================================================================
        predictions_list = []
        for pred_val, pred_date in zip(future_predictions, future_dates):
            predictions_list.append({
                "date": convert_to_serializable(pred_date),
                "predicted_value": convert_to_serializable(pred_val)
            })
        
        # 최종 결과 반환
        return {
            "metadata": {
                "model_name": config.get('modelName', 'unknown'),
                "target_column": targetColumn,
                "prediction_steps": future_steps,
                "last_known_date": convert_to_serializable(last_date),
                "method": "최소 개입 예측 (5σ 범위만 제한)",
                "historical_mean": historical_mean,
                "historical_std": historical_std,
                "learned_patterns": {
                    "workday": patterns["workday"],
                    "holiday": patterns["holiday"]
                }
            },
            "predictions": predictions_list,
            "statistics": {
                "min_predicted": convert_to_serializable(np.min(future_predictions)),
                "max_predicted": convert_to_serializable(np.max(future_predictions)),
                "mean_predicted": convert_to_serializable(np.mean(future_predictions)),
                "std_predicted": convert_to_serializable(np.std(future_predictions))
            }
        }
        
    except Exception as e:
        print(f"❌ 예측 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# 예측 결과를 데이터베이스에 저장
# ============================================================================

def save_predictions_to_db(prediction_result, target_table="usage_generation_forecast"):
    """
    예측 결과를 PostgreSQL 데이터베이스에 저장
    
    Args:
        prediction_result (dict): predict_future_stable()의 반환값
        target_table (str): 저장할 테이블명 (기본: usage_generation_forecast)
    
    Returns:
        tuple: (성공 건수, 실패 건수)
    
    저장 프로세스:
        1. 기존 동일 시점 데이터 삭제 (중복 방지)
        2. 새로운 예측값 삽입
        3. 트랜잭션으로 묶어서 all-or-nothing 보장
    
    테이블 구조:
        - time_point: 예측 시점 (datetime)
        - forecast_usage_kwh: 예측 전력 사용량 (float)
        - reg_dt: 등록 일시 (timestamp)
    """
    if prediction_result is None:
        return 0, 0
    
    try:
        engine = get_db_engine()
        predictions = prediction_result.get('predictions', [])
        
        if not predictions:
            return 0, 0
        
        print(f"\n💾 DB 저장 시작...")
        
        success_count = 0
        
        # 트랜잭션 시작
        with engine.connect() as conn:
            trans = conn.begin()

            try:
                conn.execute(text("SET timezone = 'Asia/Seoul'"))
                # 각 예측값에 대해
                for pred in predictions:
                    # 1) 기존 데이터 삭제 (동일 시점)
                    delete_query = text(f"DELETE FROM carbontwin.{target_table} WHERE time_point = :time_point")
                    conn.execute(delete_query, {"time_point": pred['date']})

                    # 2) 새로운 예측값 삽입
                    insert_query = text(f"""
                    INSERT INTO carbontwin.{target_table} (time_point, forecast_usage_kwh, reg_dt)
                    VALUES (:time_point, :forecast_value, now())
                    """)
                    
                    conn.execute(insert_query, {
                        "time_point": pred['date'],
                        "forecast_value": pred['predicted_value']
                    })
                    
                    success_count += 1
                
                # 트랜잭션 커밋
                trans.commit()
                print(f"✅ DB 저장 완료: {success_count}건")
                
            except Exception as e:
                # 오류 발생 시 롤백
                trans.rollback()
                print(f"❌ DB 저장 오류: {str(e)}")
                return success_count, len(predictions) - success_count
        
        return success_count, 0
        
    except Exception as e:
        print(f"❌ DB 연결 오류: {str(e)}")
        return 0, len(predictions) if predictions else 0

# ============================================================================
# 메인 실행 함수
# ============================================================================

def main(model_name, tablename, future_steps=672, save_to_db_flag=True, validation_days=1):
    """
    전력 사용량 예측 전체 프로세스 실행
    
    Args:
        model_name (str): 모델명 (예: usage-kwh-model-4)
        tablename (str): 데이터 테이블명 (예: lstm_input_15m_new)
        future_steps (int): 예측할 미래 스텝 수 (기본 672 = 7일)
        save_to_db_flag (bool): DB 저장 여부 (기본 True)
        validation_days (int): 검증할 최근 일수 (기본 7일)
    
    Returns:
        dict: 검증 결과와 예측 결과를 담은 딕셔너리
            - validation: 검증 결과
            - future_prediction: 미래 예측 결과
        None: 실패 시
    
    실행 순서:
        1. 모델 로드 (LSTM 모델, 스케일러, 설정)
        2. 데이터베이스에서 데이터 로드
        3. 최근 N일 데이터로 모델 검증 (정확도 측정)
        4. 미래 예측 수행
        5. 예측 결과를 데이터베이스에 저장
    """
    print("=" * 80)
    print("⚡ 전력 사용량 예측 (개선된 역정규화 방식)")
    print("=" * 80)
    
    # 1) 모델 로드
    model, scaler, config = load_trained_model(model_name)
    if model is None:
        return None
    
    # 2) 데이터 로드
    print(f"\n📊 데이터 로드 중...")
    new_data = load_new_data(tablename, config['dateColumn'], config['studyColumns'])
    if new_data is None or new_data.empty:
        return None
    
    # 🔥 데이터 충분성 체크
    seq_len = int(config['r_seqLen'])
    r_predDays = int(config.get('r_predDays', 1))
    min_required_for_validation = seq_len + (validation_days * 96) + r_predDays
    min_required_for_prediction = seq_len  # 예측만 하려면 시퀀스 길이만 있으면 됨
    
    print(f"\n📏 데이터 체크:")
    print(f"   현재 데이터: {len(new_data)}행")
    print(f"   예측 최소 요구: {min_required_for_prediction}행")
    print(f"   검증 최소 요구: {min_required_for_validation}행")
    
    # 예측조차 불가능한 경우
    if len(new_data) < min_required_for_prediction:
        print(f"\n❌ 데이터 부족: 예측 불가")
        print(f"   최소 {min_required_for_prediction}행 필요 (현재: {len(new_data)}행)")
        return None
    
    # 검증 가능 여부 판단
    validation_result = None
    if len(new_data) >= min_required_for_validation:
        print(f"\n✅ 검증 가능 → 검증 수행")
        # 3) 모델 검증
        validation_result = validate_with_actual_data(
            model, scaler, config, new_data, validation_days
        )
        
        if validation_result:
            val_accuracy = validation_result['statistics']['accuracy']
            print(f"\n✅ 검증 정확도: {val_accuracy:.2f}%")
    else:
        print(f"\n⚠️  데이터 부족: 검증 건너뛰고 예측만 수행")
        print(f"   (검증하려면 {min_required_for_validation}행 필요)")
    
    # 4) 미래 예측 수행 (검증 결과 있으면 활용, 없으면 None)
    print(f"\n🔮 미래 예측 시작 ({future_steps}스텝 = {future_steps//96}일)")
    
    future_result = predict_future_stable(
        model, scaler, config, new_data, future_steps,
        historical_mean=validation_result.get('historical_mean') if validation_result else None,
        historical_std=validation_result.get('historical_std') if validation_result else None
    )
    
    # 5) DB 저장
    if future_result and save_to_db_flag:
        success, fail = save_predictions_to_db(future_result)
        if success > 0:
            print(f"\n✅ {success}건 저장")
        if fail > 0:
            print(f"⚠️  {fail}건 저장 실패")
    
    print(f"\n{'='*80}")
    print("🎉 완료!")
    print("="*80)
    
    return {
        "validation": validation_result,
        "future_prediction": future_result
    }

# ============================================================================
# 스크립트 직접 실행 시
# ============================================================================

if __name__ == "__main__":
    try:
        # 실행 설정
        model_name = "usage_kwh_model"  # 사용할 모델명
        tablename = "lstm_input_15m"   # 데이터 테이블명
        
        print("\n" + "=" * 80)
        print("⚡ 개선된 역정규화 방식 적용")
        print("=" * 80)
        
        # 메인 함수 실행
        result = main(
            model_name=model_name,
            tablename=tablename,
            future_steps=672,      # 7일 예측 (96 * 7)
            save_to_db_flag=True,  # DB 저장 활성화
            validation_days=1      # 검증 일수 (데이터 부족 시 자동 스킵)
        )
        
        # 최종 결과 출력
        if result:
            if result.get('validation'):
                val_stats = result['validation']['statistics']
                print(f"\n{'='*80}")
                print(f"📊 최종 요약")
                print(f"{'='*80}")
                print(f"   정확도: {val_stats['accuracy']:.2f}%")
                print(f"   MAPE:   {val_stats['mape']:.2f}%")
                print(f"{'='*80}")
            elif result.get('future_prediction'):
                print(f"\n{'='*80}")
                print(f"📊 최종 요약 (검증 없음)")
                print(f"{'='*80}")
                stats = result['future_prediction']['statistics']
                print(f"   예측값 범위: {stats['min_predicted']:.2f} ~ {stats['max_predicted']:.2f} kWh")
                print(f"   예측값 평균: {stats['mean_predicted']:.2f} kWh")
                print(f"{'='*80}")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  중단")
    except Exception as e:
        print(f"\n❌ 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"\n❌ 오류: {str(e)}")
        import traceback
        traceback.print_exc()