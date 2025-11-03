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

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import joblib
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta

ENV = os.getenv('FLASK_ENV', 'local')
if ENV == 'local':
    root = "D:/work/lstm"
else:
    root = "/app/webfiles/lstm"

model_path = os.path.abspath(root + "/saved_models")
prediction_path = os.path.abspath(root + "/predictions")
os.makedirs(prediction_path, exist_ok=True)

def get_db_engine():
    connection_string = "postgresql://postgres:mapinus@10.10.10.201:5432/postgres"
    return create_engine(connection_string)

def convert_to_serializable(obj):
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

def load_new_data(tablename, dateColumn, studyColumns):
    try:
        engine = get_db_engine()
        
        query = f"""
            SELECT {studyColumns},{dateColumn}
            FROM carbontwin.{tablename}
            WHERE {dateColumn} IS NOT NULL
            ORDER BY {dateColumn} ASC
            """
        
        data = pd.read_sql_query(query, engine)
        print(f"✅ 데이터 로드: {len(data)}행")
        
        if len(data) > 0 and dateColumn in data.columns:
            data[dateColumn] = pd.to_datetime(data[dateColumn])
            min_date = data[dateColumn].min()
            max_date = data[dateColumn].max()
            print(f"   📅 기간: {min_date} ~ {max_date}")
        
        return data
        
    except Exception as e:
        print(f"❌ 데이터 로드 오류: {str(e)}")
        return None

def load_trained_model(model_name):
    try:
        model_file = os.path.join(model_path, f"{model_name}.h5")
        scaler_file = os.path.join(model_path, f"{model_name}_scaler.pkl")
        config_file = os.path.join(model_path, f"{model_name}_config.json")
        
        if not all(os.path.exists(f) for f in [model_file, scaler_file, config_file]):
            print(f"❌ 필요한 파일을 찾을 수 없습니다.")
            return None, None, None
        
        print(f"📂 모델 로드: {model_name}")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = load_model(model_file, compile=False)
            model.compile(optimizer='adam', loss='mse')
            scaler = joblib.load(scaler_file)
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print(f"✅ 모델 로드 완료")
        print(f"   - 타겟: {config['targetColumn']}")
        print(f"   - 시퀀스: {config['r_seqLen']}")
        print(f"   - 스케일러: {type(scaler).__name__}")
        
        return model, scaler, config
        
    except Exception as e:
        print(f"❌ 모델 로드 오류: {str(e)}")
        return None, None, None

def validate_with_actual_data(model, scaler, config, data, validation_days=7):
    """검증 함수 - 개선된 역정규화 방식"""
    try:
        print(f"\n{'='*80}")
        print(f"🔍 모델 검증 시작 (최근 {validation_days}일)")
        print(f"{'='*80}")
        
        dateColumn = config['dateColumn']
        studyColumns = config['studyColumns']
        targetColumn = config['targetColumn']
        seq_len = int(config['r_seqLen'])
        r_predDays = int(config.get('r_predDays', 1))
        
        study_columns_list = [col.strip() for col in studyColumns.split(',')]
        target_idx = study_columns_list.index(targetColumn)
        
        data_for_prediction = data[study_columns_list].astype(float)
        dates = pd.to_datetime(data[dateColumn])
        
        validation_points = 96 * validation_days
        validation_start_idx = len(data) - validation_points - r_predDays
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data_scaled = scaler.transform(data_for_prediction)
        
        testX, testY = [], []
        test_range = range(seq_len, len(data_scaled) - r_predDays + 1)
        
        for i in test_range:
            if i < validation_start_idx:
                continue
            testX.append(data_scaled[i - seq_len:i, :].astype(np.float32))
            testY.append(data_scaled[i + r_predDays - 1:i + r_predDays, target_idx].astype(np.float32))
        
        testX = np.array(testX, dtype=np.float32)
        testY = np.array(testY, dtype=np.float32)
        
        print(f"\n🔄 역정규화 방식: 전체 피처 벡터 활용 (스케일러 독립적)")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prediction = model.predict(testX, verbose=0)
            
            # ✅ 개선된 역정규화: 전체 피처 벡터 방식
            # 예측값 역정규화
            y_pred = []
            for i, pred_scaled in enumerate(prediction):
                # testX[i]의 마지막 스텝을 베이스로 사용
                full_scaled = testX[i, -1, :].copy()  # 마지막 타임스텝의 전체 피처
                full_scaled[target_idx] = pred_scaled[0]  # 타겟 위치에 예측값 삽입
                full_original = scaler.inverse_transform(full_scaled.reshape(1, -1))[0]
                y_pred.append(full_original[target_idx])
            y_pred = np.array(y_pred)
            
            # 실제값 역정규화
            testY_original = []
            for i, y_scaled in enumerate(testY):
                full_scaled = testX[i, -1, :].copy()  # 마지막 타임스텝의 전체 피처
                full_scaled[target_idx] = y_scaled[0]  # 타겟 위치에 실제값 삽입
                full_original = scaler.inverse_transform(full_scaled.reshape(1, -1))[0]
                testY_original.append(full_original[target_idx])
            testY_original = np.array(testY_original)
        
        eps = 9
        mask = testY_original > eps
        mape = np.mean(np.abs((y_pred[mask] - testY_original[mask]) / testY_original[mask])) * 100 if np.sum(mask) > 0 else 999.0
        
        accuracy = 100 - mape
        mae = np.mean(np.abs(y_pred - testY_original))
        rmse = np.sqrt(np.mean((y_pred - testY_original) ** 2))
        
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

def predict_future_stable(model, scaler, config, data, future_steps=672, historical_mean=None, historical_std=None):
    """
    LSTM 모델을 사용한 미래 전력 사용량 예측 함수
    
    주요 기능:
    1. 데이터에서 평일/휴일 패턴 자동 학습
    2. 요일별 맞춤형 안정화 적용
    3. 전체 피처 벡터 방식의 안전한 역정규화
    4. 시간 특성 자동 업데이트
    
    Args:
        model: 학습된 LSTM 모델
        scaler: 데이터 정규화에 사용된 스케일러 (StandardScaler, MinMaxScaler 등)
        config: 모델 설정 딕셔너리 (시퀀스 길이, 컬럼 정보 등)
        data: 과거 데이터 DataFrame
        future_steps: 예측할 미래 스텝 수 (기본값: 672 = 7일 × 96개/일)
        historical_mean: 과거 평균값 (None이면 자동 계산)
        historical_std: 과거 표준편차 (None이면 자동 계산)
    
    Returns:
        dict: 예측 결과를 담은 딕셔너리
            - metadata: 모델 정보, 예측 방법 등
            - predictions: 날짜별 예측값 리스트
            - statistics: 예측 통계 (최소, 최대, 평균, 표준편차)
    
    전제 조건:
        - 토요일, 일요일 = 휴일 (생산 거의 없음)
        - 월~금 = 평일 (정상 생산)
        - 시간대별 특성 없음 (요일만 중요)
    """
    
    def calculate_workday_holiday_patterns(data_for_prediction, dates, targetColumn):
        """
        과거 데이터에서 평일/휴일 패턴을 자동으로 학습하는 함수
        
        이 함수는 과거 데이터를 분석하여:
        1. 평일(월~금)과 휴일(토~일)로 데이터를 분리
        2. 각 그룹의 통계적 특성 계산 (평균, 표준편차, 분위수 등)
        3. 요일별 상세 정보도 함께 계산
        
        Args:
            data_for_prediction: 예측에 사용할 데이터 DataFrame
            dates: 날짜 정보가 담긴 Series
            targetColumn: 예측 대상 컬럼명 (예: 'usage_kwh')
        
        Returns:
            tuple: (patterns, weekday_details)
                - patterns: 평일/휴일 패턴 딕셔너리
                - weekday_details: 요일별 상세 정보 딕셔너리
        """
        print(f"   🔍 평일/휴일 패턴 학습 중...")
        
        # 타겟 컬럼의 값들을 numpy 배열로 추출
        target_values = data_for_prediction[targetColumn].values
        
        # 평일 (월~금) vs 휴일 (토, 일) 분리
        # weekday(): 0=월요일, 1=화요일, ..., 4=금요일, 5=토요일, 6=일요일
        weekday_mask = dates.dt.weekday < 5  # 0~4 = 월~금 (True/False 배열)
        weekend_mask = dates.dt.weekday >= 5  # 5~6 = 토~일 (True/False 배열)
        
        # 마스크를 사용하여 데이터 분리
        weekday_values = target_values[weekday_mask]  # 평일 데이터만 추출
        weekend_values = target_values[weekend_mask]  # 휴일 데이터만 추출
        
        # 평일/휴일 각각의 통계적 특성 계산
        patterns = {
            "workday": {  # 평일 패턴
                "mean": float(np.mean(weekday_values)),      # 평균
                "std": float(np.std(weekday_values)),        # 표준편차
                "median": float(np.median(weekday_values)),  # 중앙값
                "min": float(np.min(weekday_values)),        # 최소값
                "max": float(np.max(weekday_values)),        # 최대값
                "q25": float(np.percentile(weekday_values, 25)),  # 1사분위수 (25%)
                "q75": float(np.percentile(weekday_values, 75)),  # 3사분위수 (75%)
                "zero_ratio": float(np.sum(weekday_values == 0) / len(weekday_values)),  # 0값 비율
                "count": len(weekday_values)  # 데이터 개수
            },
            "holiday": {  # 휴일 패턴
                "mean": float(np.mean(weekend_values)),
                "std": float(np.std(weekend_values)),
                "median": float(np.median(weekend_values)),
                "min": float(np.min(weekend_values)),
                "max": float(np.max(weekend_values)),
                "q25": float(np.percentile(weekend_values, 25)),
                "q75": float(np.percentile(weekend_values, 75)),
                "zero_ratio": float(np.sum(weekend_values == 0) / len(weekend_values)),
                "count": len(weekend_values)
            }
        }
        
        # 각 요일별 상세 정보도 추가로 계산 (출력 및 분석용)
        weekday_names = ['월', '화', '수', '목', '금', '토', '일']
        weekday_details = {}
        
        for day_idx in range(7):  # 0~6 = 월~일
            # 특정 요일의 데이터만 추출
            day_mask = dates.dt.weekday == day_idx
            day_values = target_values[day_mask]
            
            if len(day_values) > 0:  # 해당 요일 데이터가 있는 경우
                weekday_details[day_idx] = {
                    "name": weekday_names[day_idx],  # 요일 이름
                    "mean": float(np.mean(day_values)),  # 평균
                    "std": float(np.std(day_values)),    # 표준편차
                    "zero_ratio": float(np.sum(day_values == 0) / len(day_values)),  # 0값 비율
                    "count": len(day_values),  # 데이터 개수
                    "is_workday": day_idx < 5  # 평일 여부 (월~금 = True)
                }
        
        return patterns, weekday_details
    
    def adaptive_stabilization(pred_original, next_date, patterns):
        """
        예측값을 평일/휴일 패턴에 맞게 안정화하는 함수
        
        이 함수는:
        1. 예측 날짜가 평일인지 휴일인지 판단
        2. 해당 패턴의 통계를 기준으로 예측값 안정화
        3. 평일: 평균 회귀 + 3σ 범위 제한
        4. 휴일: 엄격한 상한 제한 (생산 거의 없음)
        
        Args:
            pred_original: 모델의 원래 예측값
            next_date: 예측 날짜 (datetime)
            patterns: 평일/휴일 패턴 딕셔너리
        
        Returns:
            tuple: (안정화된 예측값, 안정화 여부, 사유, 요일명, 타입, 아이콘)
        """
        # 예측 날짜의 요일 확인 (0=월요일, 6=일요일)
        day_of_week = next_date.weekday()
        
        # 평일 여부 판단 (월~금 = True)
        is_workday = day_of_week < 5
        
        # 패턴 선택 (평일 또는 휴일)
        if is_workday:
            pattern = patterns["workday"]  # 평일 패턴 사용
            day_type = "평일"
            icon = "🏢"
        else:
            pattern = patterns["holiday"]  # 휴일 패턴 사용
            day_type = "휴일"
            icon = "🏖️"
        
        # 요일 이름 가져오기
        weekday_names = ['월', '화', '수', '목', '금', '토', '일']
        weekday_name = weekday_names[day_of_week]
        
        # 패턴의 통계값 추출
        mean = pattern["mean"]      # 평균
        std = pattern["std"]        # 표준편차
        q25 = pattern["q25"]        # 1사분위수
        q75 = pattern["q75"]        # 3사분위수
        min_val = pattern["min"]    # 최소값
        max_val = pattern["max"]    # 최대값
        
        # 안정화 추적 변수
        original_pred = pred_original  # 원본 예측값 저장
        stabilization_applied = False  # 안정화 적용 여부
        stabilization_reason = ""      # 안정화 사유
        
        # ====================================================================
        # 평일 안정화 로직
        # ====================================================================
        if is_workday:  # 평일 (월~금)
            # 1단계: 평균 회귀 (Mean Reversion)
            # 예측값이 평균에서 너무 멀리 떨어져 있으면 평균쪽으로 당김
            deviation = abs(pred_original - mean)  # 평균과의 거리
            threshold = 2.5 * std  # 임계값: 2.5 표준편차
            
            if deviation > threshold:
                # 평균과 예측값의 가중 평균으로 보정
                # alpha=0.6 → 평균에 60%, 예측값에 40% 가중치
                alpha = 0.6
                pred_original = alpha * mean + (1 - alpha) * pred_original
                stabilization_applied = True
                stabilization_reason = f"평일 평균 회귀 (편차: {deviation:.1f})"
            
            # 2단계: 극단값 제한 (3σ 범위)
            # 예측값이 정상 범위를 벗어나면 강제로 범위 내로 제한
            safe_min = max(0, mean - 3 * std)  # 하한: 평균 - 3표준편차 (음수 방지)
            safe_max = min(max_val, mean + 3 * std)  # 상한: 평균 + 3표준편차 (최대값 초과 방지)
            
            # 하한 체크
            if pred_original < safe_min:
                pred_original = safe_min
                if not stabilization_applied:
                    stabilization_applied = True
                    stabilization_reason = "평일 최소값 제한"
            
            # 상한 체크
            elif pred_original > safe_max:
                pred_original = safe_max
                if not stabilization_applied:
                    stabilization_applied = True
                    stabilization_reason = "평일 최대값 제한"
        
        # ====================================================================
        # 휴일 안정화 로직
        # ====================================================================
        else:  # 휴일 (토, 일)
            # 휴일은 생산이 거의 없으므로 엄격한 제한 적용
            
            # 1단계: 상한 제한 (평균 + 2표준편차)
            upper_limit = mean + 2 * std  # 평일보다 엄격한 기준 (2σ vs 3σ)
            
            if pred_original > upper_limit:
                # IQR(Interquartile Range) 범위로 제한
                # IQR = Q3 - Q1, 이상치 탐지에 사용되는 통계적 범위
                pred_original = np.clip(pred_original, min_val, q75)
                stabilization_applied = True
                stabilization_reason = f"휴일 제한 (상한: {upper_limit:.1f})"
            
            # 2단계: 극단값 억제
            # 예측값이 평균의 3배를 넘으면 확률적으로 낮은 값으로 대체
            if pred_original > mean * 3:
                # 최소값과 평균*1.5 사이의 랜덤값으로 대체
                pred_original = np.random.uniform(min_val, mean * 1.5)
                stabilization_applied = True
                stabilization_reason = "휴일 극단값 억제"
        
        # ====================================================================
        # 최종 안전 범위
        # ====================================================================
        # 음수 방지 (전력 사용량은 음수가 될 수 없음)
        pred_original = max(0, pred_original)
        
        return pred_original, stabilization_applied, stabilization_reason, weekday_name, day_type, icon
    
    def update_time_features(next_row, next_date, study_columns_list):
        """
        시퀀스 업데이트 시 시간 관련 특성을 동적으로 업데이트하는 함수
        
        시간이 지남에 따라 변하는 특성들을 자동으로 업데이트:
        - week_code: 요일 코드 (1~6)
        - is_weekend: 주말 여부 (0 또는 1)
        - is_workday: 평일 여부 (0 또는 1)
        - day_sin, day_cos: 요일 순환 인코딩
        
        Args:
            next_row: 업데이트할 데이터 행 (numpy array)
            next_date: 다음 예측 날짜 (datetime)
            study_columns_list: 특성 컬럼 리스트
        
        Returns:
            numpy.ndarray: 시간 특성이 업데이트된 데이터 행
        """
        # 예측 날짜의 요일 확인 (0=월요일, 6=일요일)
        day_of_week = next_date.weekday()
        
        # ====================================================================
        # 1. week_code 업데이트
        # ====================================================================
        # week_code: 요일을 숫자로 표현 (월=1, 화=2, ..., 토=6, 일=6)
        # 주말(토, 일)은 모두 6으로 처리
        if 'week_code' in study_columns_list:
            idx = study_columns_list.index('week_code')
            # day_of_week: 0(월)~6(일) → +1하면 1~7
            # min(..., 6)으로 일요일(7)도 6으로 제한
            next_row[idx] = min(day_of_week + 1, 6)
        
        # ====================================================================
        # 2. is_weekend 업데이트
        # ====================================================================
        # is_weekend: 주말(토, 일) 여부를 이진값으로 표현
        # 토요일(5), 일요일(6) = 1, 그 외 = 0
        if 'is_weekend' in study_columns_list:
            idx = study_columns_list.index('is_weekend')
            next_row[idx] = 1 if day_of_week >= 5 else 0
        
        # ====================================================================
        # 3. is_workday 업데이트
        # ====================================================================
        # is_workday: 평일(월~금) 여부를 이진값으로 표현
        # 월~금 = 1, 토~일 = 0
        # 주의: 시간대 무관! (요일만으로 판단)
        if 'is_workday' in study_columns_list:
            idx = study_columns_list.index('is_workday')
            next_row[idx] = 1 if day_of_week < 5 else 0
        
        # ====================================================================
        # 4. 순환 인코딩 (Cyclic Encoding) - 요일
        # ====================================================================
        # 요일을 sin/cos로 인코딩하여 연속성 표현
        # 예: 일요일(6)과 월요일(0)이 가깝다는 것을 수치적으로 표현
        
        # day_sin: sin(2π × 요일 / 7)
        # 월(0)→0.00, 화(1)→0.78, 수(2)→0.97, ..., 일(6)→-0.43
        if 'day_sin' in study_columns_list:
            idx = study_columns_list.index('day_sin')
            next_row[idx] = np.sin(2 * np.pi * day_of_week / 7)
        
        # day_cos: cos(2π × 요일 / 7)
        # 월(0)→1.00, 화(1)→0.62, 수(2)→-0.22, ..., 일(6)→0.90
        if 'day_cos' in study_columns_list:
            idx = study_columns_list.index('day_cos')
            next_row[idx] = np.cos(2 * np.pi * day_of_week / 7)
        
        # 순환 인코딩의 장점:
        # - 일요일(6)과 월요일(0)의 거리가 1로 표현됨
        # - 선형 인코딩(0,1,2,...,6)보다 요일 간 연속성을 더 잘 표현
        
        return next_row
    
    # ====================================================================
    # 메인 예측 로직 시작
    # ====================================================================
    
    try:
        print(f"\n{'='*80}")
        print(f"🔮 평일/휴일 기반 예측 ({future_steps}개 스텝 = {future_steps//96}일)")
        print(f"{'='*80}")
        
        # ====================================================================
        # 1단계: 설정 정보 추출
        # ====================================================================
        dateColumn = config['dateColumn']        # 날짜 컬럼명 (예: 'time_point')
        studyColumns = config['studyColumns']    # 특성 컬럼들 (CSV 문자열)
        targetColumn = config['targetColumn']    # 예측 대상 컬럼 (예: 'usage_kwh')
        seq_len = int(config['r_seqLen'])       # 시퀀스 길이 (예: 672)
        r_predDays = int(config.get('r_predDays', 1))  # 예측 스텝 (기본값: 1)
        
        # 특성 컬럼 문자열을 리스트로 변환
        # 예: "is_workday,week_code,usage_kwh" → ['is_workday', 'week_code', 'usage_kwh']
        study_columns_list = [col.strip() for col in studyColumns.split(',')]
        
        # 타겟 컬럼의 인덱스 찾기
        # 예: 'usage_kwh'가 3번째 컬럼이면 target_idx = 2 (0부터 시작)
        target_idx = study_columns_list.index(targetColumn)
        
        # ====================================================================
        # 2단계: 데이터 준비
        # ====================================================================
        # 예측에 사용할 데이터만 추출 (특성 컬럼들만)
        data_for_prediction = data[study_columns_list].astype(float)
        
        # 날짜 컬럼을 datetime 형식으로 변환
        dates = pd.to_datetime(data[dateColumn])
        
        # 마지막 날짜 저장 (예측 시작점)
        last_date = dates.iloc[-1]
        
        # ====================================================================
        # 3단계: 평일/휴일 패턴 자동 학습
        # ====================================================================
        # 과거 데이터를 분석하여 평일/휴일의 통계적 특성 추출
        patterns, weekday_details = calculate_workday_holiday_patterns(
            data_for_prediction, dates, targetColumn
        )
        
        # 평균/표준편차 기본값 설정
        # 파라미터로 전달되지 않으면 전체 데이터의 평균/표준편차 사용
        if historical_mean is None:
            historical_mean = data_for_prediction[targetColumn].mean()
        if historical_std is None:
            historical_std = data_for_prediction[targetColumn].std()
        
        # ====================================================================
        # 학습된 패턴 출력
        # ====================================================================
        print(f"\n   📊 학습된 패턴:")
        
        # 평일 패턴 출력
        print(f"      🏢 평일 (월~금):")
        print(f"         - 평균: {patterns['workday']['mean']:6.2f} kWh (±{patterns['workday']['std']:5.2f})")
        print(f"         - 범위: [{patterns['workday']['min']:.2f}, {patterns['workday']['max']:.2f}]")
        print(f"         - 0값 비율: {patterns['workday']['zero_ratio']*100:4.1f}%")
        print(f"         - 데이터 수: {patterns['workday']['count']:,}개")
        
        # 휴일 패턴 출력
        print(f"\n      🏖️ 휴일 (토, 일):")
        print(f"         - 평균: {patterns['holiday']['mean']:6.2f} kWh (±{patterns['holiday']['std']:5.2f})")
        print(f"         - 범위: [{patterns['holiday']['min']:.2f}, {patterns['holiday']['max']:.2f}]")
        print(f"         - 0값 비율: {patterns['holiday']['zero_ratio']*100:4.1f}%")
        print(f"         - 데이터 수: {patterns['holiday']['count']:,}개")
        
        # 요일별 상세 정보 출력
        print(f"\n   📅 요일별 상세:")
        for day_idx in range(7):
            if day_idx in weekday_details:
                detail = weekday_details[day_idx]
                icon = "🏢" if detail["is_workday"] else "🏖️"
                print(f"      {icon} {detail['name']}요일: {detail['mean']:6.2f} kWh "
                      f"(±{detail['std']:5.2f}) | 0값: {detail['zero_ratio']*100:4.1f}%")
        
        print(f"\n   🔄 역정규화: 전체 피처 벡터 방식")
        
        # ====================================================================
        # 4단계: 데이터 정규화
        # ====================================================================
        # 학습 시 사용한 스케일러로 데이터 정규화
        # 경고 메시지 무시 설정
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data_scaled = scaler.transform(data_for_prediction)
        
        # ====================================================================
        # 5단계: 초기 시퀀스 준비
        # ====================================================================
        # 마지막 seq_len개 데이터를 초기 시퀀스로 사용
        # 예: seq_len=672이면 마지막 672개 데이터 (7일치)
        current_sequence = data_scaled[-seq_len:, :].copy()
        
        # 예측 결과 저장 리스트
        future_predictions = []  # 예측값 리스트
        future_dates = []        # 예측 날짜 리스트
        stabilization_log = []   # 안정화 로그 리스트
        max_log = 10             # 최대 로그 개수
        
        # ====================================================================
        # 6단계: 예측 루프 (핵심 로직)
        # ====================================================================
        # future_steps번 반복하여 미래 예측
        # 각 스텝마다: 예측 → 안정화 → 시퀀스 업데이트
        for step in range(future_steps):
            # ================================================================
            # 6-1. 모델 예측 (정규화 공간)
            # ================================================================
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # 현재 시퀀스를 모델 입력 형태로 변환
                # shape: (1, seq_len, num_features)
                # 1: 배치 크기, seq_len: 시퀀스 길이, num_features: 특성 개수
                X = current_sequence.reshape(1, seq_len, -1)
                
                # LSTM 모델로 예측 (정규화된 공간에서)
                # 출력: (1, 1) 형태 → [0, 0]으로 스칼라 값 추출
                pred_scaled = model.predict(X, verbose=0)[0, 0]
                
                # ============================================================
                # 6-2. 역정규화 (전체 피처 벡터 방식)
                # ============================================================
                # 스케일러 종류에 무관한 안전한 역정규화 방법
                
                # 마지막 타임스텝의 전체 특성 벡터 복사
                # shape: (num_features,)
                full_scaled = current_sequence[-1].copy()
                
                # 타겟 위치에만 예측값 삽입
                # 다른 특성들은 그대로 유지
                full_scaled[target_idx] = pred_scaled
                
                # 전체 벡터를 역정규화
                # shape: (1, num_features) → (num_features,)
                full_original = scaler.inverse_transform(full_scaled.reshape(1, -1))[0]
                
                # 타겟 값만 추출
                pred_original = float(full_original[target_idx])
                
                # 왜 이 방식을 사용하나?
                # - scaler.mean_, scaler.scale_ 등 내부 속성에 의존하지 않음
                # - StandardScaler, MinMaxScaler, RobustScaler 등 모든 스케일러 지원
                # - 다른 특성들과의 상관관계를 고려한 정확한 역변환
            
            # ================================================================
            # 6-3. 다음 날짜 계산
            # ================================================================
            # 15분 단위로 시간 증가
            # step=0일 때 → 15분 후
            # step=1일 때 → 30분 후
            # step=95일 때 → 24시간(1일) 후
            next_date = last_date + timedelta(minutes=15 * (step + 1))
            
            # ================================================================
            # 6-4. 평일/휴일 기반 안정화
            # ================================================================
            # 예측값을 데이터 패턴에 맞게 보정
            pred_original, stabilized, reason, weekday_name, day_type, icon = adaptive_stabilization(
                pred_original, next_date, patterns
            )
            
            # 안정화가 적용되었으면 로그에 기록 (최대 max_log개까지)
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
            
            # 예측 결과 저장
            future_predictions.append(pred_original)
            future_dates.append(next_date)
            
            # ================================================================
            # 6-5. 시퀀스 업데이트 (FIFO 방식)
            # ================================================================
            # 다음 예측을 위해 시퀀스 업데이트
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # 마지막 행을 복사하여 새로운 행 생성
                next_row = data_for_prediction.iloc[-1].copy().values
                
                # 타겟 위치에 예측값 삽입
                next_row[target_idx] = pred_original
                
                # 시간 관련 특성 동적 업데이트
                # week_code, is_weekend, is_workday 등을 새로운 날짜에 맞게 업데이트
                next_row = update_time_features(next_row, next_date, study_columns_list)
                
                # 새로운 행을 정규화
                next_row_scaled = scaler.transform(next_row.reshape(1, -1))[0].astype(np.float32)
                
                # NaN/Inf 체크 (안전성)
                # 정규화 과정에서 오류가 발생하면 최근 평균으로 대체
                if np.any(np.isnan(next_row_scaled)) or np.any(np.isinf(next_row_scaled)):
                    next_row_scaled = np.mean(current_sequence[-10:], axis=0)
            
            # FIFO(First In First Out) 방식으로 시퀀스 업데이트
            # 첫 번째 행 제거, 마지막에 새 행 추가
            # Before: [t-672, t-671, ..., t-1, t]
            # After:  [t-671, t-670, ..., t, t+1]
            current_sequence = np.vstack([current_sequence[1:], next_row_scaled.reshape(1, -1)])
            
            # 진행 상황 출력 (96스텝마다 = 1일마다)
            if (step + 1) % 96 == 0:
                print(f"   ⏳ {step + 1}/{future_steps} 완료 ({(step+1)//96}일)")
        
        # ====================================================================
        # 7단계: 결과 출력 및 검증
        # ====================================================================
        
        # 안정화 로그 출력
        if stabilization_log:
            print(f"\n   ⚠️  안정화 적용 사례 (총 {len(stabilization_log)}건):")
            for log in stabilization_log[:5]:  # 처음 5개만 출력
                print(f"      {log['icon']} {log['date']} ({log['weekday']}, {log['type']}): "
                      f"{log['value']:.2f} kWh - {log['reason']}")
            if len(stabilization_log) > 5:
                print(f"      ... 외 {len(stabilization_log) - 5}건")
        
        # numpy 배열로 변환 (통계 계산 용이)
        future_predictions = np.array(future_predictions)
        
        # 예측 통계 출력
        print(f"\n📊 예측 결과:")
        print(f"   - 최소: {np.min(future_predictions):.2f} kWh")
        print(f"   - 최대: {np.max(future_predictions):.2f} kWh")
        print(f"   - 평균: {np.mean(future_predictions):.2f} kWh")
        print(f"   - 표준편차: {np.std(future_predictions):.2f} kWh")
        
        # ====================================================================
        # 8단계: 예측 품질 검증
        # ====================================================================
        # 예측된 평일/휴일 평균을 학습 데이터와 비교
        
        workday_predictions = []   # 평일 예측값 리스트
        holiday_predictions = []   # 휴일 예측값 리스트
        
        # 예측값을 평일/휴일로 분리
        for pred_val, pred_date in zip(future_predictions, future_dates):
            if pred_date.weekday() < 5:  # 평일
                workday_predictions.append(pred_val)
            else:  # 휴일
                holiday_predictions.append(pred_val)
        
        print(f"\n   📅 예측된 평일/휴일 평균 (vs 학습 데이터):")
        
        # 평일 비교
        if workday_predictions:
            pred_workday_avg = np.mean(workday_predictions)
            actual_workday_avg = patterns["workday"]["mean"]
            diff = pred_workday_avg - actual_workday_avg
            diff_pct = (diff / actual_workday_avg * 100) if actual_workday_avg > 0 else 0
            print(f"      🏢 평일: {pred_workday_avg:6.2f} kWh "
                  f"(학습: {actual_workday_avg:6.2f}, 차이: {diff:+6.2f} / {diff_pct:+5.1f}%)")
        
        # 휴일 비교
        if holiday_predictions:
            pred_holiday_avg = np.mean(holiday_predictions)
            actual_holiday_avg = patterns["holiday"]["mean"]
            diff = pred_holiday_avg - actual_holiday_avg
            diff_pct = (diff / actual_holiday_avg * 100) if actual_holiday_avg > 0 else 0
            print(f"      🏖️ 휴일: {pred_holiday_avg:6.2f} kWh "
                  f"(학습: {actual_holiday_avg:6.2f}, 차이: {diff:+6.2f} / {diff_pct:+5.1f}%)")
        
        # ====================================================================
        # 9단계: 결과 포맷팅
        # ====================================================================
        # 예측 결과를 딕셔너리 리스트로 변환 (JSON 직렬화 가능)
        predictions_list = []
        for pred_val, pred_date in zip(future_predictions, future_dates):
            predictions_list.append({
                "date": convert_to_serializable(pred_date),
                "predicted_value": convert_to_serializable(pred_val)
            })
        
        # ====================================================================
        # 10단계: 최종 결과 반환
        # ====================================================================
        return {
            "metadata": {
                "model_name": config.get('modelName', 'unknown'),
                "target_column": targetColumn,
                "prediction_steps": future_steps,
                "last_known_date": convert_to_serializable(last_date),
                "method": "평일/휴일 기반 적응형 안정화",
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
        # 예외 발생 시 오류 메시지 출력
        print(f"❌ 예측 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def save_predictions_to_db(prediction_result, target_table="usage_generation_forecast"):
    if prediction_result is None:
        return 0, 0
    
    try:
        engine = get_db_engine()
        predictions = prediction_result.get('predictions', [])
        
        if not predictions:
            return 0, 0
        
        print(f"\n💾 DB 저장 시작...")
        
        success_count = 0
        
        with engine.connect() as conn:
            trans = conn.begin()
            
            try:
                for pred in predictions:
                    delete_query = text(f"DELETE FROM carbontwin.{target_table} WHERE time_point = :time_point")
                    conn.execute(delete_query, {"time_point": pred['date']})
                    
                    insert_query = text(f"""
                    INSERT INTO carbontwin.{target_table} (time_point, forecast_usage_kwh, reg_dt)
                    VALUES (:time_point, :forecast_value, CURRENT_TIMESTAMP)
                    """)
                    
                    conn.execute(insert_query, {
                        "time_point": pred['date'],
                        "forecast_value": pred['predicted_value']
                    })
                    
                    success_count += 1
                
                trans.commit()
                print(f"✅ DB 저장 완료: {success_count}건")
                
            except Exception as e:
                trans.rollback()
                print(f"❌ DB 저장 오류: {str(e)}")
                return success_count, len(predictions) - success_count
        
        return success_count, 0
        
    except Exception as e:
        print(f"❌ DB 연결 오류: {str(e)}")
        return 0, len(predictions) if predictions else 0

def main(model_name, tablename, future_steps=672, save_to_db_flag=True, validation_days=7):
    print("=" * 80)
    print("⚡ 전력 사용량 예측 (개선된 역정규화 방식)")
    print("=" * 80)
    
    model, scaler, config = load_trained_model(model_name)
    if model is None:
        return None
    
    print(f"\n📊 데이터 로드 중...")
    new_data = load_new_data(tablename, config['dateColumn'], config['studyColumns'])
    if new_data is None or new_data.empty:
        return None
    
    validation_result = validate_with_actual_data(model, scaler, config, new_data, validation_days)
    
    if validation_result:
        val_accuracy = validation_result['statistics']['accuracy']
        print(f"\n✅ 검증 정확도: {val_accuracy:.2f}%")
        
        future_result = predict_future_stable(
            model, scaler, config, new_data, future_steps,
            historical_mean=validation_result.get('historical_mean'),
            historical_std=validation_result.get('historical_std')
        )
        
        if future_result and save_to_db_flag:
            success, fail = save_predictions_to_db(future_result)
            if success > 0:
                print(f"\n✅ {success}건 저장")
    
    print(f"\n{'='*80}")
    print("🎉 완료!")
    print("="*80)
    
    return {"validation": validation_result, "future_prediction": future_result}

if __name__ == "__main__":
    try:
        model_name = "usage-kwh-model-2"
        tablename = "lstm_input_15m_new"
        
        print("\n" + "=" * 80)
        print("⚡ 개선된 역정규화 방식 적용")
        print("=" * 80)
        
        result = main(
            model_name=model_name,
            tablename=tablename,
            future_steps=672,
            save_to_db_flag=True,
            validation_days=7
        )
        
        if result and result.get('validation'):
            val_stats = result['validation']['statistics']
            print(f"\n{'='*80}")
            print(f"📊 최종 요약")
            print(f"{'='*80}")
            print(f"   정확도: {val_stats['accuracy']:.2f}%")
            print(f"   MAPE:   {val_stats['mape']:.2f}%")
            print(f"{'='*80}")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  중단")
    except Exception as e:
        print(f"\n❌ 오류: {str(e)}")
        import traceback
        traceback.print_exc()