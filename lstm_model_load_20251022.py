# -*- coding: utf-8 -*-
"""
Title   : EPS 임계값 필터링이 적용된 LSTM 예측 스크립트
Author  : 주성중 / (주)맵인어스
Description: 
    - 학습된 LSTM 모델로 신규 데이터 예측 수행
    - EPS 임계값 기반 예측 신뢰도 필터링 추가
    - 미래값 예측 기능 포함
    - PostgreSQL DB 저장 기능
Version : 2.4
Date    : 2025-10-22
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import joblib
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# 환경 설정 블록
# -----------------------------------------------------------------------------
ENV = os.getenv('FLASK_ENV', 'local')
if ENV == 'local':
    # 개발(로컬) 환경일 때의 루트 경로
    root = "D:/work/lstm"
else:
    # 배포(컨테이너 등) 환경일 때의 루트 경로
    root = "/app/webfiles/lstm"

# 모델과 예측 결과를 저장/불러올 디렉토리 경로
model_path = os.path.abspath(root + "/saved_models")
prediction_path = os.path.abspath(root + "/predictions")
os.makedirs(prediction_path, exist_ok=True)  # 디렉토리가 없으면 생성

# -----------------------------------------------------------------------------
# 🔥 EPS 임계값 설정 (전역 변수)
# -----------------------------------------------------------------------------
# 이 값은 학습 과정에서 사용한 임계값과 동일하게 맞추는 것이 권장됩니다.
# EPS: Very small energy outputs를 무시하기 위한 임계값 (kWh 단위 예시)
PREDICTION_EPS_THRESHOLD = 0  # 0.1 kWh 이하는 신뢰도 낮음으로 간주

# -----------------------------------------------------------------------------
# DB 연결 함수
# -----------------------------------------------------------------------------
def get_db_engine():
    """PostgreSQL 데이터베이스 연결 엔진 생성

    반환:
        sqlalchemy Engine 객체
    주의:
        - connection_string은 환경별 비밀번호/호스트에 따라 수정 필요
        - 운영 환경에서는 비밀번호를 코드에 직접 두지 말고 환경변수/시크릿 매니저 사용 권장
    """
    connection_string = "postgresql://postgres:mapinus@10.10.10.201:5432/postgres"
    return create_engine(connection_string)

def convert_to_serializable(obj):
    """NumPy 및 Pandas의 특수 타입을 JSON 직렬화 가능한 Python 기본 타입으로 변환

    사용처:
        - 예측 결과를 JSON으로 반환하거나 DB에 저장할 때 직렬화 문제 방지
    지원 타입:
        - np.ndarray -> list
        - np.integer / np.floating -> int / float
        - pandas Timestamp / datetime -> ISO 8601 문자열
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

# -----------------------------------------------------------------------------
# 신규 데이터 로드
# -----------------------------------------------------------------------------
def load_new_data(tablename, dateColumn, studyColumns, start_date=None, end_date=None, days_limit=7):
    """PostgreSQL DB에서 예측할 신규 데이터를 로드

    파라미터:
        tablename: DB 테이블 이름 (카타로그 없이 테이블명만)
        dateColumn: 시간 컬럼명 (예: 'time_point')
        studyColumns: 예측에 사용되는 컬럼들의 문자열 (콤마 구분)
        start_date, end_date: 기간 필터 (문자열 또는 None)
        days_limit: 사용되진 않지만 기본 파라미터로 남겨둠 (호출부 호환성 유지)

    반환값:
        pandas DataFrame (성공) 또는 None (실패)
    예외/주의:
        - 쿼리에서 사용되는 컬럼명은 SQL 인젝션에 취약할 수 있으니
          외부 입력을 그대로 넣는 경우 검증 필요
        - 네트워크/DB 연결 실패 시 None 리턴
    """
    try:
        engine = get_db_engine()
        
        if start_date is None and end_date is None:
            # 날짜 필터가 없을 때: 전체 데이터(정렬 포함) 조회
            query = f"""
            SELECT {studyColumns},{dateColumn}
            FROM carbontwin.{tablename}
            WHERE {dateColumn} IS NOT NULL
            ORDER BY {dateColumn} ASC
            """
        else:
            # start/end가 지정된 경우 조건 생성
            where_conditions = [f"{dateColumn} IS NOT NULL"]
            if start_date:
                where_conditions.append(f"{dateColumn} >= '{start_date}'")
            if end_date:
                where_conditions.append(f"{dateColumn} <= '{end_date}'")
            
            query = f"""
            SELECT {studyColumns},{dateColumn}
            FROM carbontwin.{tablename}
            WHERE {' AND '.join(where_conditions)}
            ORDER BY {dateColumn} ASC
            """
        
        # pandas의 read_sql_query로 결과를 DataFrame으로 반환
        data = pd.read_sql_query(query, engine)
        print(f"✅ 신규 데이터 로드 완료: {len(data)}행")
        
        # 로드된 데이터의 기간 정보 출력(디버그 목적)
        if len(data) > 0 and dateColumn in data.columns:
            min_date = pd.to_datetime(data[dateColumn]).min()
            max_date = pd.to_datetime(data[dateColumn]).max()
            print(f"   📅 데이터 기간: {min_date} ~ {max_date}")
        
        return data
        
    except Exception as e:
        # DB/쿼리 오류가 발생하면 None을 반환
        print(f"❌ 데이터 로드 오류: {str(e)}")
        return None

# -----------------------------------------------------------------------------
# 모델 로드
# -----------------------------------------------------------------------------
def load_trained_model(model_name):
    """저장된 LSTM 모델, 스케일러, 설정 파일을 로드

    파일 구성(관례):
        - 모델: {model_name}.h5
        - 스케일러: {model_name}_scaler.pkl (joblib으로 저장된 sklearn 스케일러)
        - 설정: {model_name}_config.json (json 포맷, 필수 키: studyColumns, targetColumn, dateColumn, r_seqLen 등)

    반환값:
        (model, scaler, config) 또는 (None, None, None) on error

    주의:
        - load_model에서 compile=False로 로드한 뒤 compile 호출함(호환성 보장)
        - 스케일러/설정 파일이 없으면 None 리턴
    """
    try:
        model_file = os.path.join(model_path, f"{model_name}.h5")
        scaler_file = os.path.join(model_path, f"{model_name}_scaler.pkl")
        config_file = os.path.join(model_path, f"{model_name}_config.json")
        
        if not all(os.path.exists(f) for f in [model_file, scaler_file, config_file]):
            print(f"❌ 필요한 파일을 찾을 수 없습니다.")
            return None, None, None
        
        print(f"📂 모델 로드 중: {model_name}")
        
        # Keras 모델 로드 (컴파일 옵션은 나중에 설정)
        model = load_model(model_file, compile=False)
        model.compile(optimizer='adam', loss='mse')  # 예측용으로 기본 컴파일
        
        # 스케일러와 config 로드
        scaler = joblib.load(scaler_file)
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # studyColumns 문자열을 리스트로 변환 (공백 제거)
        study_cols_list = [col.strip() for col in config['studyColumns'].split(',')]
        
        print(f"✅ 모델 로드 완료")
        print(f"   - 타겟 컬럼: {config['targetColumn']}")
        print(f"   - 시퀀스 길이: {config['r_seqLen']}")
        print(f"   - EPS 임계값: {PREDICTION_EPS_THRESHOLD}")
        
        return model, scaler, config
        
    except Exception as e:
        print(f"❌ 모델 로드 오류: {str(e)}")
        return None, None, None

# -----------------------------------------------------------------------------
# 🔥 EPS 기반 예측 신뢰도 분석 함수
# -----------------------------------------------------------------------------
def analyze_prediction_reliability(predictions, eps_threshold=PREDICTION_EPS_THRESHOLD):
    """
    예측값의 신뢰도를 EPS 임계값 기반으로 분석

    설명:
        - predictions 배열을 eps_threshold와 비교하여 신뢰 가능한 예측/신뢰 불가 예측으로 분류
        - 신뢰 가능한 예측들에 대한 기본 통계(min/max/mean/median/std) 계산
        - 신뢰 불가 예측들에 대한 기본 통계(min/max/mean/median) 계산

    반환값:
        dict 형태의 분석 결과:
            {
                "eps_threshold": eps_threshold,
                "total_predictions": total_count,
                "reliable_predictions": reliable_count,
                "unreliable_predictions": unreliable_count,
                "reliability_ratio": ratio,
                "reliable_indices": [...],
                "unreliable_indices": [...],
                "reliable_statistics": {...} or None,
                "unreliable_statistics": {...} or None
            }

    주의:
        - total_count가 0일 경우 ratio는 0으로 처리
        - 통계값은 float로 변환하여 JSON 시리얼라이즈가 가능하도록 함
    """
    predictions = np.array(predictions)
    
    # EPS 임계값 기반 분류 (열 기준)
    reliable_mask = predictions > eps_threshold
    unreliable_mask = ~reliable_mask
    
    reliable_count = np.sum(reliable_mask)
    unreliable_count = np.sum(unreliable_mask)
    total_count = len(predictions)
    
    analysis = {
        "eps_threshold": eps_threshold,
        "total_predictions": total_count,
        "reliable_predictions": reliable_count,
        "unreliable_predictions": unreliable_count,
        "reliability_ratio": reliable_count / total_count if total_count > 0 else 0,
        "reliable_indices": np.where(reliable_mask)[0].tolist(),
        "unreliable_indices": np.where(unreliable_mask)[0].tolist()
    }
    
    # 신뢰 가능한 예측값 통계
    if reliable_count > 0:
        reliable_values = predictions[reliable_mask]
        analysis["reliable_statistics"] = {
            "min": float(np.min(reliable_values)),
            "max": float(np.max(reliable_values)),
            "mean": float(np.mean(reliable_values)),
            "median": float(np.median(reliable_values)),
            "std": float(np.std(reliable_values))
        }
    else:
        analysis["reliable_statistics"] = None
    
    # 신뢰할 수 없는 예측값 통계
    if unreliable_count > 0:
        unreliable_values = predictions[unreliable_mask]
        analysis["unreliable_statistics"] = {
            "min": float(np.min(unreliable_values)),
            "max": float(np.max(unreliable_values)),
            "mean": float(np.mean(unreliable_values)),
            "median": float(np.median(unreliable_values))
        }
    else:
        analysis["unreliable_statistics"] = None
    
    return analysis

# -----------------------------------------------------------------------------
# 🔥 EPS 필터링을 적용한 예측값 출력 함수
# -----------------------------------------------------------------------------
def print_predictions_with_eps_filter(predictions, dates, eps_threshold=PREDICTION_EPS_THRESHOLD):
    """
    EPS 임계값 기반으로 필터링된 예측값을 테이블 형식으로 출력

    동작:
        - analyze_prediction_reliability를 호출해 통계 및 인덱스를 얻음
        - 신뢰 가능한 예측값(최대 20개)과 신뢰 불가 예측값(최대 10개)을 표 형태로 출력
        - 각 예측값에 대해 간단한 '신뢰도' 텍스트(높음/보통/낮음)를 표시

    출력은 디버그/모니터링 용도로 사용되며, 실제 저장/응답은 별도 로직에서 처리
    """
    predictions = np.array(predictions)
    
    # 신뢰도 분석
    reliability = analyze_prediction_reliability(predictions, eps_threshold)
    
    print(f"\n📊 EPS 임계값 기반 예측 신뢰도 분석")
    print(f"{'='*90}")
    print(f"   🎯 EPS 임계값: {eps_threshold}")
    print(f"   📈 전체 예측: {reliability['total_predictions']}개")
    print(f"   ✅ 신뢰 가능 ({eps_threshold} 초과): {reliability['reliable_predictions']}개 "
          f"({reliability['reliability_ratio']*100:.1f}%)")
    print(f"   ⚠️  신뢰 불가 ({eps_threshold} 이하): {reliability['unreliable_predictions']}개 "
          f"({(1-reliability['reliability_ratio'])*100:.1f}%)")
    
    if reliability["reliable_statistics"]:
        stats = reliability["reliable_statistics"]
        print(f"\n   ✅ 신뢰 가능 예측값 통계:")
        print(f"      - 범위: {stats['min']:.4f} ~ {stats['max']:.4f}")
        print(f"      - 평균: {stats['mean']:.4f}")
        print(f"      - 중앙값: {stats['median']:.4f}")
        print(f"      - 표준편차: {stats['std']:.4f}")
    
    if reliability["unreliable_statistics"]:
        stats = reliability["unreliable_statistics"]
        print(f"\n   ⚠️  신뢰 불가 예측값 통계:")
        print(f"      - 범위: {stats['min']:.4f} ~ {stats['max']:.4f}")
        print(f"      - 평균: {stats['mean']:.4f}")
    
    print(f"{'='*90}")
    
    # 신뢰 가능한 예측값만 출력 (최대 20개)
    reliable_indices = reliability['reliable_indices']
    
    if len(reliable_indices) > 0:
        print(f"\n✅ 신뢰 가능한 예측값 (EPS > {eps_threshold}) - 최대 20개")
        print(f"{'='*90}")
        print(f"{'인덱스':>6} {'날짜/시간':<25} {'예측값':>12} {'신뢰도':>10}")
        print(f"{'-'*90}")
        
        display_count = min(20, len(reliable_indices))
        for i in range(display_count):
            idx = reliable_indices[i]
            date_str = dates[idx].strftime('%Y-%m-%d %H:%M:%S') if hasattr(dates[idx], 'strftime') else str(dates[idx])
            pred_val = predictions[idx]
            # 간단한 등급화: EPS * 10을 초과하면 '높음', 아니면 '보통'
            confidence = "높음" if pred_val > eps_threshold * 10 else "보통"
            
            print(f"{idx:>6} {date_str:<25} {pred_val:>12.4f} {confidence:>10}")
        
        if len(reliable_indices) > 20:
            print(f"... ({len(reliable_indices) - 20}개 더 있음)")
        
        print(f"{'='*90}")
    else:
        # 신뢰 가능한 예측값이 아예 없을 때의 안내문
        print(f"\n⚠️  신뢰 가능한 예측값이 없습니다!")
        print(f"   💡 모델 재학습을 권장합니다.")
    
    # 신뢰 불가 예측값도 일부 출력 (처음 10개만)
    unreliable_indices = reliability['unreliable_indices']
    
    if len(unreliable_indices) > 0:
        print(f"\n⚠️  신뢰 불가 예측값 (EPS ≤ {eps_threshold}) - 처음 10개")
        print(f"{'='*90}")
        print(f"{'인덱스':>6} {'날짜/시간':<25} {'예측값':>12} {'상태':>10}")
        print(f"{'-'*90}")
        
        display_count = min(10, len(unreliable_indices))
        for i in range(display_count):
            idx = unreliable_indices[i]
            date_str = dates[idx].strftime('%Y-%m-%d %H:%M:%S') if hasattr(dates[idx], 'strftime') else str(dates[idx])
            pred_val = predictions[idx]
            
            print(f"{idx:>6} {date_str:<25} {pred_val:>12.4f} {'⚠️ 낮음':>10}")
        
        if len(unreliable_indices) > 10:
            print(f"... ({len(unreliable_indices) - 10}개 더 있음)")
        
        print(f"{'='*90}")

# -----------------------------------------------------------------------------
# 🔥 EPS 필터링이 적용된 미래값 예측 함수
# -----------------------------------------------------------------------------
def predict_future_with_eps(model, scaler, config, new_data, future_steps=None, 
                            eps_threshold=PREDICTION_EPS_THRESHOLD, 
                            apply_filter=True):
    """
    EPS 임계값 필터링이 적용된 미래값 예측

    주요 로직 요약:
        1) 입력 데이터의 마지막 seq_len 구간을 가져와 시퀀스를 구성
        2) 루프를 돌며 한 스텝씩 예측 (auto-regressive 방식)
        3) 예측 스텝마다 스케일링 역변환을 통해 원단위 예측값을 얻음
        4) EPS 임계값과 시간대(주간/야간)에 따라 필터링 적용
           - pred_original <= eps_threshold -> 0으로 설정 (노이즈 제거)
           - 야간(6시 미만 또는 18시 이후)에는 원본의 10%만 적용 (야간 보수적 적용)
        5) 필터링된 값을 시퀀스에 반영하여 다음 스텝 예측에 사용
        6) 예측 결과와 신뢰도(0~1)를 구성하여 반환

    파라미터:
        model: Keras 학습된 모델
        scaler: 학습 때 사용한 스케일러 (mean_, scale_ 속성 필요)
        config: 모델 설정(dict) - 반드시 'dateColumn','studyColumns','targetColumn','r_seqLen' 포함
        new_data: 예측에 사용할 최신 데이터(DataFrame)
        future_steps: 예측할 스텝 수 (기본 값이 None이면 672로 설정)
        eps_threshold: EPS 임계값 (float)
        apply_filter: True이면 필터링 적용(작업 기본값)

    반환값:
        dict 형태의 예측 결과 (예: predictions 리스트, 통계, 신뢰도 분석 등)
    """
    try:
        dateColumn = config['dateColumn']
        studyColumns = config['studyColumns']
        targetColumn = config['targetColumn']
        seq_len = int(config['r_seqLen'])
        pred_days = int(config['r_predDays'])
        
        # 기본값: 7일치 (15분 간격 가정 시 7*96 = 672)
        if future_steps is None:
            future_steps = 672  # 7일 = 7 * 96 (15분 간격)
        
        study_columns_list = [col.strip() for col in studyColumns.split(',')]
        target_idx = study_columns_list.index(targetColumn)
        
        # 마지막 시간 정보 추출: new_data의 마지막 행의 dateColumn 사용
        if dateColumn in new_data.columns:
            last_date = pd.to_datetime(new_data[dateColumn].iloc[-1])
        else:
            last_date = datetime.now()
        
        print(f"\n🔮 EPS 필터링 미래값 예측 시작...")
        print(f"   - 시퀀스 길이: {seq_len}개")
        print(f"   - 예측 스텝: {future_steps}개")
        print(f"   - EPS 임계값: {eps_threshold}")
        print(f"   - 필터링 적용: {'예' if apply_filter else '아니오'}")
        
        # 예측에 사용할 입력 부분만 float 타입으로 변환
        data_for_prediction = new_data[study_columns_list].astype(float)
        
        # 입력 데이터가 시퀀스 길이보다 작으면 에러
        if len(data_for_prediction) < seq_len:
            raise ValueError(f"데이터 부족: {len(data_for_prediction)}개 (최소 {seq_len}개 필요)")
        
        # 정규화 (scaler를 사용하여 학습 시와 동일한 변환 적용)
        data_scaled = scaler.transform(data_for_prediction)
        
        # 시간 간격 계산: 마지막 두 행의 차이로 시간 간격을 추정 (없으면 15분 가정)
        if dateColumn in new_data.columns and len(new_data) > 1:
            dates = pd.to_datetime(new_data[dateColumn])
            time_delta = (dates.iloc[-1] - dates.iloc[-2])
        else:
            time_delta = pd.Timedelta(minutes=15)
        
        # 현재 시퀀스를 마지막 seq_len 데이터로 초기화
        current_sequence = data_scaled[-seq_len:].copy()
        
        # 결과 저장용 리스트들
        future_predictions = []
        future_predictions_raw = []  # 필터링 전 원본값
        future_dates = []
        prediction_confidence = []
        
        # 기준값(baseline) 계산: 최근 100개 중 양수값의 중앙값 사용(없으면 eps 사용)
        recent_data = data_for_prediction[targetColumn].tail(100)
        recent_positive = recent_data[recent_data > eps_threshold]
        baseline = recent_positive.median() if len(recent_positive) > 0 else eps_threshold
        
        print(f"   📊 예측 기준값: {baseline:.4f}")
        
        # 예측 루프: 각 스텝마다 예측하고 시퀀스를 업데이트
        for step in range(future_steps):
            next_date = last_date + time_delta * (step + 1)
            hour = next_date.hour
            
            # 모델 입력 형태 맞춤: (1, seq_len, feature_count)
            input_data = current_sequence.reshape(1, seq_len, len(study_columns_list))
            pred_scaled = model.predict(input_data, verbose=0)[0, 0]
            
            # 역정규화: scaler.scale_[target_idx]와 mean_[target_idx] 이용
            pred_original = pred_scaled * scaler.scale_[target_idx] + scaler.mean_[target_idx]
            
            # 원본 예측값 저장 (필터 적용 전)
            future_predictions_raw.append(pred_original)
            
            # 🔥 EPS 필터링 적용 로직:
            # - eps 이하이면 0으로 강제
            # - 낮에는 그대로, 밤에는 10%만 적용 (노이즈 억제)
            if apply_filter:
                if pred_original <= eps_threshold:
                    pred_filtered = 0.0
                else:
                    if 6 <= hour < 18:
                        pred_filtered = pred_original
                    else:
                        # 야간 보수적 적용: 원본의 10%만 사용
                        pred_filtered = max(0, pred_original * 0.1)
            else:
                # 필터링을 사용하지 않을 경우 음수는 0으로 보정
                pred_filtered = max(0, pred_original)
            
            # 신뢰도 계산: baseline과 비교하여 0~1 범위로 스케일링 (단순화된 방식)
            if pred_filtered > eps_threshold:
                confidence = min(1.0, pred_filtered / (baseline * 2))
            else:
                confidence = 0.0
            
            # 결과들에 추가
            future_predictions.append(pred_filtered)
            future_dates.append(next_date)
            prediction_confidence.append(confidence)
            
            # 다음 스텝을 위해 시퀀스에 새 샘플을 추가
            # - new_point는 마지막 row의 복사본을 사용해 다른 feature는 유지
            # - target 컬럼만 새 예측값으로 대체 (스케일링 후 반영)
            new_point = current_sequence[-1].copy()
            new_point_scaled = (pred_filtered - scaler.mean_[target_idx]) / scaler.scale_[target_idx]
            new_point[target_idx] = new_point_scaled
            
            # 슬라이딩 윈도우: 첫 행 제거하고 새 행 추가
            current_sequence = np.vstack([current_sequence[1:], new_point])
            
            # 진행 로그 (디버그 목적)
            if (step + 1) % 100 == 0:
                print(f"   ⏳ 진행: {step + 1}/{future_steps} 스텝 완료")
        
        print(f"✅ 예측 완료!")
        
        # 신뢰도 분석 (EPS 기준)
        reliability = analyze_prediction_reliability(future_predictions, eps_threshold)
        
        # 요약 출력
        print(f"\n📊 예측 결과 요약:")
        print(f"   - 전체 예측: {len(future_predictions)}개")
        print(f"   - 신뢰 가능: {reliability['reliable_predictions']}개 "
              f"({reliability['reliability_ratio']*100:.1f}%)")
        print(f"   - 신뢰 불가: {reliability['unreliable_predictions']}개")
        print(f"   - 예측값 범위: {min(future_predictions):.4f} ~ {max(future_predictions):.4f}")
        
        # 테이블 형태로 주요 결과 출력 (콘솔)
        print_predictions_with_eps_filter(future_predictions, future_dates, eps_threshold)
        
        # 반환용 결과 딕셔너리 구성
        future_result = {
            "model_name": config['modelName'],
            "target_column": targetColumn,
            "prediction_type": "future_with_eps_filter",
            "base_date": last_date.isoformat(),
            "sequence_length": seq_len,
            "future_steps": future_steps,
            "eps_threshold": eps_threshold,
            "filter_applied": apply_filter,
            "reliability_analysis": reliability,
            "baseline_value": float(baseline),
            "predictions": []
        }
        
        # 각 스텝 결과를 리스트에 순차적으로 추가
        for i, (date, pred, pred_raw, conf) in enumerate(
            zip(future_dates, future_predictions, future_predictions_raw, prediction_confidence)):
            future_result["predictions"].append({
                "step": i + 1,
                "date": date.isoformat(),
                "predicted_value": convert_to_serializable(pred),
                "predicted_value_raw": convert_to_serializable(pred_raw),
                "confidence": convert_to_serializable(conf),
                "hour": date.hour,
                "is_reliable": pred > eps_threshold,
                "is_daytime": 6 <= date.hour < 18
            })
        
        # 전체 통계: numpy를 사용해 간단히 계산하고 직렬화 준비
        future_result["statistics"] = {
            "min_predicted": convert_to_serializable(np.min(future_predictions)),
            "max_predicted": convert_to_serializable(np.max(future_predictions)),
            "mean_predicted": convert_to_serializable(np.mean(future_predictions)),
            "median_predicted": convert_to_serializable(np.median(future_predictions)),
            "std_predicted": convert_to_serializable(np.std(future_predictions))
        }
        
        return future_result
        
    except Exception as e:
        # 예측 중 예외 발생 시 스택 트레이스 출력 후 None 반환
        print(f"❌ 미래값 예측 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# -----------------------------------------------------------------------------
# 🔥 EPS 필터링 적용한 DB 저장 함수
# -----------------------------------------------------------------------------
def save_predictions_to_db_with_eps(prediction_result, target_table="solar_generation_forecast", 
                                    only_reliable=False):
    """
    미래 예측 결과를 PostgreSQL DB에 저장 (EPS 필터링 옵션)

    파라미터:
        prediction_result: predict_future_with_eps의 반환 dict
        target_table: 저장 대상 테이블명 (carbontwin.<target_table> 사용)
        only_reliable: True이면 is_reliable == True인 예측만 저장

    동작:
        - 기존 동일 time_point 레코드는 DELETE로 제거(중복 방지)
        - INSERT로 새 레코드 추가 (time_point, forecast_solar_kwh, reg_dt)
        - 트랜잭션으로 묶어 중간 오류 시 롤백

    반환값:
        (success_count, fail_count)
    주의:
        - 실제 테이블 스키마(칼럼명)가 다르면 INSERT문 수정 필요
        - 시간 포맷은 ISO8601 문자열로 전달되므로 DB의 time_point 칼럼 타입에 맞게 변환될 것
    """
    if prediction_result is None:
        print("❌ 저장할 예측 결과가 없습니다.")
        return 0, 0
    
    try:
        engine = get_db_engine()
        predictions = prediction_result.get('predictions', [])
        
        if not predictions:
            print("❌ 예측 데이터가 비어있습니다.")
            return 0, 0
        
        # only_reliable 옵션에 따라 필터링
        if only_reliable:
            predictions = [p for p in predictions if p.get('is_reliable', False)]
            print(f"\n📊 신뢰 가능한 예측만 저장: {len(predictions)}건")
        
        print(f"\n💾 예측 결과 DB 저장 시작...")
        print(f"   - 대상 테이블: carbontwin.{target_table}")
        print(f"   - 저장할 데이터: {len(predictions)}건")
        
        success_count = 0
        fail_count = 0
        
        # DB 커넥션과 트랜잭션 처리
        with engine.connect() as conn:
            trans = conn.begin()
            
            try:
                for pred in predictions:
                    time_point = pred['date']
                    forecast_value = pred['predicted_value']
                    
                    # 중복 제거: 동일 time_point인 경우 삭제(정책)
                    delete_query = text(f"""
                    DELETE FROM carbontwin.{target_table}
                    WHERE time_point = :time_point
                    """)
                    
                    conn.execute(delete_query, {"time_point": time_point})
                    
                    # 삽입: 기본 컬럼명 사용 (필요시 수정)
                    insert_query = text(f"""
                    INSERT INTO carbontwin.{target_table} 
                        (time_point, forecast_solar_kwh, reg_dt)
                    VALUES 
                        (:time_point, :forecast_value, CURRENT_TIMESTAMP)
                    """)
                    
                    conn.execute(
                        insert_query,
                        {
                            "time_point": time_point,
                            "forecast_value": forecast_value
                        }
                    )
                    
                    success_count += 1
                    
                    # 대량 삽입시 진행 로그 출력(디버그/모니터링)
                    if success_count % 100 == 0:
                        print(f"   ⏳ 진행: {success_count}/{len(predictions)} 건")
                
                trans.commit()
                
                print(f"✅ DB 저장 완료!")
                print(f"   - 성공: {success_count}건")
                
            except Exception as e:
                trans.rollback()
                print(f"❌ DB 저장 중 오류 (롤백됨): {str(e)}")
                return success_count, len(predictions) - success_count
        
        return success_count, fail_count
        
    except Exception as e:
        print(f"❌ DB 연결 오류: {str(e)}")
        return 0, len(predictions) if predictions else 0

# -----------------------------------------------------------------------------
# 메인 실행 함수
# -----------------------------------------------------------------------------
def main(model_name=None, tablename=None, save_to_db=True, only_reliable=False, 
         eps_threshold=PREDICTION_EPS_THRESHOLD, apply_filter=True):
    """메인 실행 함수

    동작 요약:
        1. 모델/스케일러/설정 로드
        2. DB에서 신규 데이터 로드
        3. predict_future_with_eps로 미래 예측 수행
        4. save_predictions_to_db_with_eps로 DB에 저장 (옵션)
        5. 예외/오류 발생 시 적절히 메시지 출력

    반환값:
        predict_future_with_eps가 반환한 결과 dict 또는 None
    """
    print("=" * 70)
    print("🔮 EPS 필터링 적용 LSTM 예측 시스템")
    print("=" * 70)
    
    # 모델 로드
    model, scaler, config = load_trained_model(model_name)
    
    if model is None:
        return None
    
    if tablename is None:
        tablename = "lstm_input_15m_new"
    
    print(f"\n📊 데이터베이스에서 데이터 로드 중...")
    # load_new_data의 days_limit 파라미터는 기본값으로 호출
    new_data = load_new_data(tablename, config['dateColumn'], config['studyColumns'], days_limit=7)
    
    if new_data is None or new_data.empty:
        print("❌ 예측할 데이터가 없습니다.")
        return None
    
    # 예측 스텝 기본: 7일
    future_steps = 672  # 7일
    
    print(f"\n🔮 미래값 예측 수행")
    print(f"   - 예측 스텝: {future_steps}개")
    print(f"   - EPS 임계값: {eps_threshold}")
    print(f"   - 필터링 적용: {'예' if apply_filter else '아니오'}")
    
    # 실제 예측 호출
    future_result = predict_future_with_eps(
        model, scaler, config, new_data, future_steps,
        eps_threshold, apply_filter
    )
    
    # 예측 결과가 있고 DB 저장 옵션이 켜져 있으면 저장 수행
    if future_result and save_to_db:
        success, fail = save_predictions_to_db_with_eps(
            future_result, 
            only_reliable=only_reliable
        )
        
        if success > 0:
            print(f"\n✅ 총 {success}건의 예측 결과가 DB에 저장되었습니다.")
            if only_reliable:
                print(f"   💡 신뢰 가능한 예측만 저장되었습니다.")
        if fail > 0:
            print(f"⚠️  {fail}건의 저장 실패")
    
    print(f"\n{'='*70}")
    print("🎉 예측 완료!")
    print("="*70)
    
    return future_result

# -----------------------------------------------------------------------------
# 프로그램 시작점
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    """
    EPS 필터링 적용 예측 스크립트 실행

    사용법:
        python lstm_predict_with_eps.py

    실행시 제공되는 옵션:
        - 사용자가 콘솔에서 모드를 선택하고 EPS 값 입력 가능
        - 기본적으로 모델명과 테이블명은 스크립트 내부의 기본값을 사용
    """
    try:
        model_name = "solar-hybrid-seq-2-test-20251017-test-no-add-test"
        tablename = "lstm_input_15m_new"
        
        print("\n" + "=" * 80)
        print("🔍 실행 모드 선택")
        print("=" * 80)
        print("\n1. EPS 필터링 적용 예측 (권장)")
        # 사용자가 입력하지 않으면 기본값 "1" 사용
        
        # EPS 임계값 설정: 입력이 없으면 전역값 사용
        eps_threshold = PREDICTION_EPS_THRESHOLD;
        
        print(f"\n⚙️  설정:")
        print(f"   - EPS 임계값: {eps_threshold}")
        
        # EPS 필터링 적용, 전체 저장
        print(f"   - 필터링: 적용")
        print(f"   - DB 저장: 전체")

        main(
                model_name=model_name,
                tablename=tablename,
                save_to_db=True,
                only_reliable=False,
                eps_threshold=eps_threshold,
                apply_filter=True
            )
            
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()


# -----------------------------------------------------------------------------
# 📚 사용 예시 및 가이드 (문서화 주석)
# -----------------------------------------------------------------------------
"""
🎯 EPS 임계값 필터링의 장점:

1. **신뢰도 높은 예측만 선별**
   - EPS 이하의 불안정한 예측값 제거
   - 야간(0 근처) 예측의 노이즈 감소

2. **학습 코드와 일관성 유지**
   - 학습 시 MAPE 계산에 사용한 동일한 임계값 적용
   - 평가 기준과 예측 기준 일치

3. **데이터 품질 향상**
   - DB에 저장되는 예측값의 신뢰도 향상
   - 후속 분석 시 더 안정적인 데이터 사용

4. **유연한 설정**
   - eps_threshold 값 조정 가능
   - only_reliable 옵션으로 저장 범위 선택
   - apply_filter 옵션으로 필터링 on/off

📊 권장 EPS 임계값:
   - 태양광 발전량: 0.1 ~ 1.0 kWh
   - 전력 사용량: 1.0 ~ 5.0 kWh
   - 온도 예측: 0.5 ~ 1.0°C

💡 사용 팁:
   1. 먼저 eps_threshold=0.1로 테스트
   2. 신뢰도 분석 결과 확인
   3. 필요시 임계값 조정
   4. only_reliable=True로 신뢰 가능한 예측만 저장

⚠️  주의사항:
   - 임계값이 너무 높으면 대부분의 예측이 제외됨
   - 임계값이 너무 낮으면 노이즈가 많은 예측 포함
   - 모델 재학습 시 동일한 임계값 사용 권장

🔧 추가 개선 방향:
   1. 시간대별 임계값 적용 (주간/야간 다르게)
   2. 신뢰도 점수 기반 가중 평균
   3. 이상치 탐지 알고리즘 결합
   4. 앙상블 예측과 결합
"""
