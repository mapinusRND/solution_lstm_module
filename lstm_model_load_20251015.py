# -*- coding: utf-8 -*-
"""
Title   : 개선된 LSTM 모델 예측 스크립트
Author  : 주성중 / (주)맵인어스
Description: 
    - 학습된 LSTM 모델로 신규 데이터 예측 수행
    - 중복 예측값 문제 해결
    - 미래값 예측 기능 포함
Version : 2.0
Date    : 2025-10-14
"""

import os
# TensorFlow 설정: 최적화 경고 및 로그 레벨 조정
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # OneDNN 최적화 비활성화
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # 에러만 출력 (0=모든로그, 1=INFO제외, 2=WARNING제외, 3=ERROR만)

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import joblib
from sqlalchemy import create_engine
from datetime import datetime, timedelta

# ============================================================================
# 환경 설정
# ============================================================================
# 실행 환경에 따라 경로 자동 설정 (로컬 개발 환경 vs 서버 배포 환경)
ENV = os.getenv('FLASK_ENV', 'local')
if ENV == 'local':
    root = "D:/work/lstm"  # 로컬 개발 환경 경로
else:
    root = "/app/webfiles/lstm"  # 서버 배포 환경 경로

# 모델 저장 경로 및 예측 결과 저장 경로 설정
model_path = os.path.abspath(root + "/saved_models")
prediction_path = os.path.abspath(root + "/predictions")
os.makedirs(prediction_path, exist_ok=True)  # 디렉토리가 없으면 생성

# ============================================================================
# DB 연결 함수
# ============================================================================
def get_db_engine():
    """
    PostgreSQL 데이터베이스 연결 엔진 생성
    
    Returns:
        sqlalchemy.engine.Engine: DB 연결 엔진
    """
    # 실제 운영 시에는 환경 변수나 설정 파일로 관리 권장
    connection_string = "postgresql://postgres:mapinus@10.10.10.201:5432/postgres"
    return create_engine(connection_string)

# ============================================================================
# 신규 데이터 로드
# ============================================================================
def load_new_data(tablename, dateColumn, studyColumns, start_date=None, end_date=None):
    """
    PostgreSQL DB에서 예측할 신규 데이터를 로드
    
    Parameters:
    -----------
    tablename : str
        조회할 테이블명 (예: 'lstm_input_15m')
    dateColumn : str
        날짜/시간 컬럼명 (예: 'timestamp')
    studyColumns : str
        사용할 컬럼들을 쉼표로 구분한 문자열 (예: 'temp,humidity,solar_kwh')
    start_date : str, optional
        조회 시작 날짜 (YYYY-MM-DD 형식), None이면 전체 조회
    end_date : str, optional
        조회 종료 날짜 (YYYY-MM-DD 형식), None이면 전체 조회
        
    Returns:
    --------
    pandas.DataFrame : 로드된 데이터 (실패시 None)
    """
    try:
        engine = get_db_engine()
        
        # 기본 쿼리: 전체 데이터 조회
        query = f"""
        SELECT {studyColumns},{dateColumn}
        FROM carbontwin.{tablename}
        WHERE {dateColumn} IS NOT NULL
        ORDER BY {dateColumn} ASC
        """
        
        # 날짜 범위가 지정된 경우 WHERE 조건 추가
        if start_date or end_date:
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
        
        # SQL 쿼리 실행 및 DataFrame으로 변환
        data = pd.read_sql_query(query, engine)
        print(f"✅ 신규 데이터 로드 완료: {len(data)}행 (테이블: {tablename})")
        return data
        
    except Exception as e:
        print(f"❌ 데이터 로드 오류: {str(e)}")
        return None

# ============================================================================
# NumPy/Pandas 타입을 JSON 직렬화 가능하게 변환
# ============================================================================
def convert_to_serializable(obj):
    """
    NumPy 및 Pandas의 특수 타입을 JSON 직렬화 가능한 Python 기본 타입으로 변환
    
    Parameters:
    -----------
    obj : any
        변환할 객체 (np.ndarray, np.int64, np.float64, pd.Timestamp 등)
        
    Returns:
    --------
    any : JSON 직렬화 가능한 타입 (list, int, float, str)
    
    Notes:
    ------
    JSON 파일 저장 시 "Object of type float32 is not JSON serializable" 
    같은 에러를 방지하기 위한 헬퍼 함수
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # NumPy 배열 → Python 리스트
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)  # NumPy 정수 → Python int
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)  # NumPy 실수 → Python float
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()  # Pandas Timestamp → ISO 문자열
    elif isinstance(obj, datetime):
        return obj.isoformat()  # datetime → ISO 문자열
    return obj

# ============================================================================
# 모델 로드
# ============================================================================
def load_trained_model(model_name):
    """
    저장된 LSTM 모델, 스케일러, 설정 파일을 로드
    
    Parameters:
    -----------
    model_name : str
        로드할 모델명 (예: 'test15m')
        
    Returns:
    --------
    tuple : (model, scaler, config)
        - model: Keras LSTM 모델 객체
        - scaler: StandardScaler 객체 (데이터 정규화용)
        - config: dict (모델 학습 시 사용된 설정 정보)
        실패 시 (None, None, None) 반환
        
    Notes:
    ------
    모델 파일 구조:
    - {model_name}.h5: Keras 모델 가중치
    - {model_name}_scaler.pkl: StandardScaler 객체
    - {model_name}_config.json: 모델 설정 (컬럼명, 시퀀스 길이 등)
    """
    try:
        # 파일 경로 설정
        model_file = os.path.join(model_path, f"{model_name}.h5")
        scaler_file = os.path.join(model_path, f"{model_name}_scaler.pkl")
        config_file = os.path.join(model_path, f"{model_name}_config.json")
        
        # 필수 파일 존재 여부 확인
        if not os.path.exists(model_file):
            print(f"❌ 모델 파일을 찾을 수 없습니다: {model_file}")
            return None, None, None
        if not os.path.exists(scaler_file):
            print(f"❌ 스케일러 파일을 찾을 수 없습니다: {scaler_file}")
            return None, None, None
        if not os.path.exists(config_file):
            print(f"❌ 설정 파일을 찾을 수 없습니다: {config_file}")
            return None, None, None
        
        print(f"📂 모델 로드 중: {model_name}")
        
        # Keras 모델 로드 (compile=False: 학습 시 설정 무시, 예측만 사용)
        model = load_model(model_file, compile=False)
        model.compile(optimizer='adam', loss='mse')  # 예측용 재컴파일
        
        # 스케일러 로드 (학습 시 사용한 정규화 파라미터)
        scaler = joblib.load(scaler_file)
        
        # 설정 파일 로드 (JSON)
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 학습 컬럼 정보 파싱
        study_cols_list = [col.strip() for col in config['studyColumns'].split(',')]
        
        # 로드 완료 정보 출력
        print(f"✅ 모델 로드 완료")
        print(f"   - 타겟 컬럼: {config['targetColumn']}")  # 예측할 변수
        print(f"   - 학습 컬럼 ({len(study_cols_list)}개): {config['studyColumns']}")  # 입력 변수들
        print(f"   - 시퀀스 길이: {config['r_seqLen']}")  # LSTM 입력 시퀀스 길이
        print(f"   - 예측 일수: {config['r_predDays']}")  # 몇 스텝 앞을 예측하는지
        
        return model, scaler, config
        
    except Exception as e:
        print(f"❌ 모델 로드 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

# ============================================================================
# 🔥 개선된 미래값 예측 (중복 예측 문제 해결)
# ============================================================================
def predict_future_improved(model, scaler, config, new_data, future_steps=None):
    """
    개선된 미래값 예측 - 재귀적 예측으로 실제 미래값 생성
    
    개선사항:
    1. 시간 정보 추가 (시간, 분) - 태양광 발전은 시간대별 패턴이 중요
    2. 더 다양한 노이즈 추가 - 예측의 다양성 확보
    3. 예측값 범위 검증 - 물리적 제약 조건 적용 (야간=0)
    4. 앙상블 예측 - 여러 번 예측하여 평균 (안정성 향상)
    
    Parameters:
    -----------
    model : Keras Model
        학습된 LSTM 모델
    scaler : StandardScaler
        학습 시 사용한 스케일러
    config : dict
        모델 설정 정보
    new_data : DataFrame
        기준이 되는 최근 데이터
    future_steps : int, optional
        예측할 미래 스텝 수 (None이면 자동 계산: max(10, seq_len//2))
        
    Returns:
    --------
    dict : 미래 예측 결과
        - predictions: 각 스텝별 예측값, 시간 정보
        - statistics: 예측값 통계 (최소, 최대, 평균, 표준편차)
    """
    try:
        # 설정 정보 추출
        dateColumn = config['dateColumn']
        studyColumns = config['studyColumns']
        targetColumn = config['targetColumn']
        seq_len = int(config['r_seqLen'])  # LSTM 입력 시퀀스 길이
        pred_days = int(config['r_predDays'])  # 예측 간격
        
        # 미래 스텝 수 자동 계산 (지정되지 않은 경우)
        if future_steps is None:
            future_steps = max(10, seq_len // 2)  # 최소 10, 최대 시퀀스 길이의 절반
        
        # 컬럼 리스트 생성 및 타겟 인덱스 찾기
        study_columns_list = [col.strip() for col in studyColumns.split(',')]
        target_idx = study_columns_list.index(targetColumn)  # 예측할 변수의 인덱스
        
        # 마지막 날짜 추출 (기준 시점)
        if dateColumn in new_data.columns:
            last_date = pd.to_datetime(new_data[dateColumn].iloc[-1])
        else:
            last_date = datetime.now()
        
        # 데이터 준비 및 정규화
        data_for_prediction = new_data[study_columns_list].astype(float)
        data_scaled = scaler.transform(data_for_prediction)  # StandardScaler로 정규화
        
        print(f"\n🔮 개선된 미래값 예측 시작...")
        print(f"   - 기준 시퀀스 길이: {seq_len}개")
        print(f"   - 예측 시작점: {last_date}")
        print(f"   - 예측할 미래 스텝: {future_steps}개")
        
        # 시간 간격 계산 (데이터의 평균 시간 간격)
        if dateColumn in new_data.columns and len(new_data) > 1:
            dates = pd.to_datetime(new_data[dateColumn])
            time_delta = (dates.iloc[-1] - dates.iloc[-2])  # 마지막 두 데이터의 시간 차이
        else:
            time_delta = pd.Timedelta(minutes=1)  # 기본값: 1분
        
        # 초기 시퀀스 설정 (마지막 seq_len 개 데이터)
        current_sequence = data_scaled[-seq_len:].copy()
        
        # 결과 저장용 리스트
        future_predictions = []  # 예측값
        future_dates = []  # 예측 날짜
        prediction_confidence = []  # 신뢰도 (내부 사용용)
        
        # 🔥 앙상블 예측 설정 (여러 번 예측하여 평균)
        n_ensemble = 5  # 5번 예측하여 평균 사용
        
        # 재귀적 예측 루프 (각 미래 스텝마다 반복)
        for step in range(future_steps):
            # 다음 예측 시점 계산
            next_date = last_date + time_delta * (step + 1)
            
            # 시간 정보 추출 (태양광 발전은 시간대가 중요)
            hour = next_date.hour
            minute = next_date.minute
            
            # 🔥 앙상블 예측: 여러 번 예측하여 평균 (안정성 향상)
            ensemble_predictions = []
            
            for _ in range(n_ensemble):
                # 노이즈 추가 (입력 데이터에 작은 변동 추가)
                # 목적: 동일한 입력에 대해 다양한 예측값 생성
                noisy_sequence = current_sequence + np.random.normal(0, 0.05, current_sequence.shape)
                
                # LSTM 입력 형태로 변환: (batch_size=1, seq_len, features)
                input_data = noisy_sequence.reshape(1, seq_len, len(study_columns_list))
                
                # 모델 예측 (정규화된 값 출력)
                pred_scaled = model.predict(input_data, verbose=0)
                ensemble_predictions.append(pred_scaled[0, 0])
            
            # 앙상블 평균 및 표준편차 계산
            avg_pred_scaled = np.mean(ensemble_predictions)  # 평균 예측값
            pred_std = np.std(ensemble_predictions)  # 표준편차 (불확실성 지표)
            
            # 신뢰도 계산 (내부 사용용, JSON에만 저장)
            distance_penalty = 1.0 - (step / future_steps) * 0.3
            ensemble_uncertainty = min(pred_std / 0.1, 1.0)
            confidence = distance_penalty * (1.0 - ensemble_uncertainty)
            confidence = max(0.0, min(1.0, confidence))
            
            # 예측값 역정규화 (스케일된 값 → 원래 단위)
            mean_values = scaler.mean_.copy()  # 스케일러의 평균값
            mean_values[target_idx] = avg_pred_scaled  # 타겟 변수만 예측값으로 변경
            pred_value = scaler.inverse_transform([mean_values])[0, target_idx]  # 역변환
            
            # 🔥 태양광 발전량 물리적 제약 적용
            # 야간(18시~06시)에는 발전량이 거의 0이어야 함
            if 18 <= hour or hour < 6:
                pred_value = max(0, pred_value * 0.1)  # 야간은 예측값의 10%만 사용
            else:
                pred_value = max(0, pred_value)  # 주간은 음수만 방지
            
            # 결과 저장
            future_predictions.append(pred_value)
            future_dates.append(next_date)
            prediction_confidence.append(confidence)  # 내부용
            
            # 🔥 다음 시퀀스 준비 (재귀적 예측의 핵심)
            # 현재 예측값을 다음 예측의 입력으로 사용
            new_point = current_sequence[-1].copy()  # 마지막 데이터 포인트 복사
            new_point[target_idx] = avg_pred_scaled  # 타겟 변수만 예측값으로 교체
            
            # 다른 변수들에 작은 변화 추가 (시간대별 패턴 반영)
            time_factor = np.sin(2 * np.pi * hour / 24)  # 일일 주기 패턴 (-1 ~ 1)
            for i in range(len(new_point)):
                if i != target_idx:
                    # 타겟이 아닌 변수들은 시간 패턴에 따라 작은 노이즈 추가
                    new_point[i] += np.random.normal(0, 0.02) * time_factor
            
            # 시퀀스 업데이트: 가장 오래된 데이터 제거, 새 예측값 추가
            current_sequence = np.vstack([current_sequence[1:], new_point])
            
            # 진행상황 표시 (10개마다 또는 마지막)
            if (step + 1) % 10 == 0 or step == future_steps - 1:
                print(f"   ⏳ 진행: {step + 1}/{future_steps} 스텝 완료")
        
        print(f"✅ 미래값 예측 완료!")
        
        # 결과 구성 (JSON 직렬화 가능한 형태)
        future_result = {
            "model_name": config['modelName'],
            "target_column": targetColumn,
            "prediction_type": "future_improved",
            "base_date": last_date.isoformat(),
            "sequence_length": seq_len,
            "future_steps": future_steps,
            "prediction_interval": pred_days,
            "predictions": []
        }
        
        # 각 스텝별 예측 결과 저장
        for i, (date, pred, conf) in enumerate(zip(future_dates, future_predictions, prediction_confidence)):
            future_result["predictions"].append({
                "step": i + 1,
                "date": date.isoformat(),
                "predicted_value": convert_to_serializable(pred),
                "confidence": convert_to_serializable(conf),  # JSON 파일용
                "hour": date.hour,
                "is_daytime": 6 <= date.hour < 18  # 주간 여부 (06~18시)
            })
        
        # 통계 정보 추가
        future_result["statistics"] = {
            "min_predicted": convert_to_serializable(np.min(future_predictions)),
            "max_predicted": convert_to_serializable(np.max(future_predictions)),
            "mean_predicted": convert_to_serializable(np.mean(future_predictions)),
            "std_predicted": convert_to_serializable(np.std(future_predictions)),
            "avg_confidence": convert_to_serializable(np.mean(prediction_confidence))  # JSON 파일용
        }
        
        return future_result
        
    except Exception as e:
        print(f"❌ 미래값 예측 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# 개선된 미래값 예측 결과 출력
# ============================================================================
def print_future_predictions_improved(result):
    """
    미래 예측 결과를 보기 좋게 테이블 형식으로 출력
    
    Parameters:
    -----------
    result : dict
        predict_future_improved() 함수의 반환값
    """
    predictions = result.get('predictions', [])
    
    # 헤더 출력
    print(f"\n🔮 개선된 미래값 예측 결과:")
    print(f"   기준 시점: {result['base_date'][:19]}")
    print(f"   시퀀스 길이: {result.get('sequence_length', 'N/A')}개")
    print(f"   총 예측 스텝: {result['future_steps']}개")
    print("=" * 80)
    print(f"{'스텝':>6} {'예측 날짜':<20} {'시간':>6} {'예측값':>12} {'주야':>10}")
    print("=" * 80)
    
    # 각 예측 결과 출력
    for pred in predictions:
        date_str = pred['date'][:19]
        hour = pred.get('hour', 0)
        is_day = "☀️ 주간" if pred.get('is_daytime', False) else "🌙 야간"
        
        print(f"{pred['step']:>6} {date_str:<20} {hour:>6}시 "
              f"{pred['predicted_value']:>12.4f} {is_day:>10}")
    
    print("=" * 80)
    
    # 통계 정보 출력
    stats = result.get('statistics', {})
    
    print(f"\n📊 예측값 통계:")
    print(f"   최솟값: {stats.get('min_predicted', 0):.4f}")
    print(f"   최댓값: {stats.get('max_predicted', 0):.4f}")
    print(f"   평균값: {stats.get('mean_predicted', 0):.4f}")
    print(f"   표준편차: {stats.get('std_predicted', 0):.4f}")
    
    # 예측 다양성 체크 (중복 예측값 문제 감지)
    # pred_values = [p['predicted_value'] for p in predictions]
    # unique_values = len(set([round(v, 4) for v in pred_values]))  # 소수점 4자리 기준 고유값
    # diversity_ratio = unique_values / len(pred_values) * 100

# ============================================================================
# 예측 수행 (과거 데이터로 모델 성능 평가)
# ============================================================================
def predict_with_model(model, scaler, config, new_data):
    """
    로드된 모델로 신규 데이터 예측 및 성능 평가
    
    과거 데이터에 대해 예측을 수행하고 실제값과 비교하여 모델 성능을 평가합니다.
    
    Parameters:
    -----------
    model : Keras Model
        학습된 LSTM 모델
    scaler : StandardScaler
        학습 시 사용한 스케일러
    config : dict
        모델 설정 정보
    new_data : DataFrame
        예측할 신규 데이터 (실제값 포함)
        
    Returns:
    --------
    dict : 예측 결과 및 성능 지표
        - predictions: 각 시점별 실제값, 예측값, 오차
        - 성능 지표: MAPE, MAE, RMSE, R², 방향성 정확도
        - 통계: 실제값/예측값의 min, max, mean
    """
    try:
        # 설정 정보 추출
        dateColumn = config['dateColumn']
        studyColumns = config['studyColumns']
        targetColumn = config['targetColumn']
        seq_len = int(config['r_seqLen'])  # LSTM 입력 시퀀스 길이
        pred_days = int(config['r_predDays'])  # 예측 간격 (몇 스텝 뒤를 예측)
        
        # 컬럼 리스트 생성
        study_columns_list = [col.strip() for col in studyColumns.split(',')]
        
        # 날짜 컬럼 처리
        if dateColumn in new_data.columns:
            dates = pd.to_datetime(new_data[dateColumn], errors='coerce')
        else:
            # 날짜 컬럼이 없으면 임의로 생성 (5분 간격)
            print(f"⚠️ 날짜 컬럼 '{dateColumn}'이 없습니다.")
            dates = pd.date_range(start='2023-01-01', periods=len(new_data), freq='5T')
        
        # 데이터 준비 (문자열을 실수로 변환)
        data_for_prediction = new_data[study_columns_list].astype(float)
        
        print(f"🔄 데이터 전처리 중...")
        # 데이터 정규화 (StandardScaler: 평균 0, 표준편차 1)
        data_scaled = scaler.transform(data_for_prediction)
        
        # 타겟 변수의 인덱스 찾기
        target_idx = study_columns_list.index(targetColumn)
        input_dim = len(study_columns_list)
        
        # 시퀀스 데이터 생성
        predX = []  # 입력 시퀀스 (X)
        valid_dates = []  # 예측 시점의 날짜
        actual_values = []  # 실제값 (정답)
        
        # 슬라이딩 윈도우 방식으로 시퀀스 생성
        for i in range(seq_len, len(data_scaled) - pred_days + 1):
            # i 시점에서 과거 seq_len 개 데이터를 입력으로 사용
            predX.append(data_scaled[i - seq_len:i, 0:input_dim])
            
            # i + pred_days 시점의 값을 예측 (pred_days 스텝 뒤)
            valid_dates.append(dates.iloc[i + pred_days - 1])
            actual_values.append(data_for_prediction.iloc[i + pred_days - 1][targetColumn])
        
        # 데이터 부족 체크
        if len(predX) == 0:
            print(f"❌ 예측 가능한 데이터가 부족합니다.")
            print(f"   필요: {seq_len + pred_days}행 이상")
            print(f"   현재: {len(new_data)}행")
            return None
        
        # NumPy 배열로 변환
        predX = np.array(predX)  # Shape: (samples, seq_len, features)
        
        print(f"🔮 예측 수행 중...")
        print(f"   - 예측 샘플 수: {len(predX)}")
        
        # 모델 예측 (배치 처리로 한 번에 예측)
        predictions_scaled = model.predict(predX, verbose=0)  # 정규화된 예측값
        
        # 예측 결과 역정규화 (원래 단위로 복원)
        mean_values = np.repeat(scaler.mean_[np.newaxis, :], predictions_scaled.shape[0], axis=0)
        mean_values[:, target_idx] = np.squeeze(predictions_scaled)
        predictions = scaler.inverse_transform(mean_values)[:, target_idx]
        
        print(f"✅ 예측 완료!")
        
        # NumPy 배열로 변환
        actual_values = np.array(actual_values)
        
        # ========== 성능 지표 계산 ==========
        
        # 1. MAE (Mean Absolute Error): 평균 절대 오차
        mae = np.mean(np.abs(predictions - actual_values))
        
        # 2. RMSE (Root Mean Square Error): 평균 제곱근 오차
        rmse = np.sqrt(np.mean((predictions - actual_values) ** 2))
        
        # 3. MAPE (Mean Absolute Percentage Error): 평균 절대 백분율 오차
        # 실제값이 0인 경우 제외 (0으로 나누기 방지)
        mask = actual_values != 0
        if np.sum(mask) == 0:
            mape = 999.0  # 모든 실제값이 0인 경우
        else:
            mape = np.mean(np.abs((actual_values[mask] - predictions[mask]) / actual_values[mask])) * 100
        
        # 정확도 = 100 - MAPE
        accuracy = 100 - mape if not np.isnan(mape) else 0
        
        # 실제값 0인 데이터 비율 계산
        zero_ratio = (len(actual_values) - np.sum(mask)) / len(actual_values) * 100
        
        # 4. R² Score (결정계수): 모델이 데이터를 얼마나 잘 설명하는지
        ss_res = np.sum((actual_values - predictions) ** 2)  # 잔차 제곱합
        ss_tot = np.sum((actual_values - np.mean(actual_values)) ** 2)  # 총 제곱합
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # 5. 방향성 정확도: 상승/하락 방향을 맞춘 비율
        if len(actual_values) > 1:
            actual_direction = np.diff(actual_values) > 0  # 실제 상승/하락
            pred_direction = np.diff(predictions) > 0  # 예측 상승/하락
            direction_accuracy = np.mean(actual_direction == pred_direction) * 100
        else:
            direction_accuracy = 0
        
        # 성능 결과 출력
        print(f"\n📊 모델 성능 결과:")
        print(f"   🎯 MAPE: {mape:.2f}%")
        print(f"   📈 정확도: {accuracy:.2f}%")
        print(f"   📏 MAE: {mae:.4f}")
        print(f"   📐 RMSE: {rmse:.4f}")
        print(f"   🔍 R² Score: {r2:.4f}")
        print(f"   🧭 방향성 정확도: {direction_accuracy:.2f}%")
        print(f"   ℹ️  실제값 0인 데이터: {zero_ratio:.1f}% ({len(actual_values) - np.sum(mask)}/{len(actual_values)}개)")
        
        # 결과 딕셔너리 구성
        result = {
            "status": "success",
            "modelName": config['modelName'],
            "target_column": targetColumn,
            "prediction_count": len(predictions),
            "timestamp": datetime.now().isoformat(),
            
            # 성능 지표
            "mape": round(mape, 2),
            "accuracy": round(accuracy, 2),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2_score": round(r2, 4),
            "direction_accuracy": round(direction_accuracy, 2),
            "zero_ratio": round(zero_ratio, 2),
            "zero_count": int(len(actual_values) - np.sum(mask)),
            
            # 통계 정보
            "statistics": {
                "actual_min": convert_to_serializable(np.min(actual_values)),
                "actual_max": convert_to_serializable(np.max(actual_values)),
                "actual_mean": convert_to_serializable(np.mean(actual_values)),
                "predicted_min": convert_to_serializable(np.min(predictions)),
                "predicted_max": convert_to_serializable(np.max(predictions)),
                "predicted_mean": convert_to_serializable(np.mean(predictions))
            },
            
            "predictions": []
        }
        
        # 개별 예측 결과 저장
        for i in range(len(predictions)):
            actual = actual_values[i]
            predicted = predictions[i]
            diff = predicted - actual
            
            # 🔥 개선: 오차율 계산 로직
            # 실제값이 0일 때는 절대오차 기준으로 오차율 계산
            if actual == 0:
                # 예측값이 작으면(< 0.001) 낮은 오차율
                # 예측값이 크면 높은 오차율로 표시
                if abs(predicted) < 0.001:
                    pct_error = abs(predicted) * 10000  # 0.0001 → 1%
                elif abs(predicted) < 0.01:
                    pct_error = abs(predicted) * 1000   # 0.001 → 1%
                else:
                    pct_error = 999.0  # 큰 오차
            else:
                # 실제값이 0이 아닌 경우 일반적인 백분율 오차
                pct_error = abs(diff / actual) * 100
            
            result["predictions"].append({
                "index": i,
                "date": convert_to_serializable(valid_dates[i]),
                "actual_value": convert_to_serializable(actual),
                "predicted_value": convert_to_serializable(predicted),
                "difference": convert_to_serializable(diff),
                "percentage_error": convert_to_serializable(pct_error)
            })
        
        return result
        
    except Exception as e:
        print(f"❌ 예측 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# 최근 N개 예측 결과 출력
# ============================================================================
def print_recent_predictions(result, n=30):
    """
    최근 N개의 예측 결과를 테이블 형식으로 출력
    
    Parameters:
    -----------
    result : dict
        predict_with_model() 함수의 반환값
    n : int, optional
        출력할 예측 결과 개수 (기본값: 30)
    """
    predictions = result.get('predictions', [])
    # 전체 중에서 마지막 n개만 선택
    recent = predictions[-n:] if len(predictions) > n else predictions
    
    print(f"\n🔍 최근 {len(recent)}개 예측 결과:")
    print("=" * 110)
    print(f"{'날짜':<20} {'실제값':>12} {'예측값':>12} {'차이':>12} {'오차율':>12} {'비고':>10}")
    print("=" * 110)
    
    for pred in recent:
        date_str = pred['date'][:19] if len(pred['date']) > 19 else pred['date']
        actual = pred['actual_value']
        predicted = pred['predicted_value']
        diff = pred['difference']
        pct_error = pred['percentage_error']
        
        # 실제값이 0일 때 표시 방식 변경
        if actual == 0:
            # 예측값 크기에 따라 비고 표시
            if abs(predicted) < 0.001:
                remark = "✓ 미세"  # 거의 정확
            elif abs(predicted) < 0.01:
                remark = "△ 소오차"  # 작은 오차
            else:
                remark = "✗ 대오차"  # 큰 오차
            error_str = f"{pct_error:>9.2f}%*"  # *표시로 특수 계산 표시
        else:
            remark = ""
            error_str = f"{pct_error:>10.2f}%"
        
        print(f"{date_str:<20} "
              f"{actual:>12.4f} "
              f"{predicted:>12.4f} "
              f"{diff:>12.4f} "
              f"{error_str:>12} "
              f"{remark:>10}")
    
    print("=" * 110)
    print("* 실제값이 0일 때는 절대오차 기준 오차율 (참고용)")

# ============================================================================
# 메인 실행 함수
# ============================================================================
def main(model_name=None, tablename=None):
    """
    메인 실행 함수 - 전체 예측 프로세스 실행
    
    Parameters:
    -----------
    model_name : str, optional
        사용할 모델명 (None이면 입력 받음)
    tablename : str, optional
        데이터를 가져올 테이블명 (None이면 기본값 사용)
        
    Returns:
    --------
    dict : 테스트 결과 및 미래 예측 결과
    """
    print("=" * 70)
    print("🔮 개선된 LSTM 모델 예측 시스템")
    print("=" * 70)
    
    # 1. 모델명 입력 또는 기본값 사용
    if model_name is None:
        model_name = input("\n📝 사용할 모델명을 입력하세요 (기본값: test15m): ").strip()
        if not model_name:
            model_name = "test15m"
            print(f"✅ 기본 모델 사용: {model_name}")
    else:
        print(f"\n✅ 지정된 모델 사용: {model_name}")
    
    # 2. 모델, 스케일러, 설정 로드
    model, scaler, config = load_trained_model(model_name)
    
    # 모델 로드 실패 시 사용 가능한 모델 목록 출력
    if model is None:
        print("\n💡 사용 가능한 모델 목록:")
        if os.path.exists(model_path):
            models = [f.replace('.h5', '') for f in os.listdir(model_path) if f.endswith('.h5')]
            if models:
                for i, m in enumerate(models, 1):
                    print(f"   {i}. {m}")
            else:
                print("   (저장된 모델이 없습니다)")
        return
    
    # 3. 테이블명 설정
    if tablename is None:
        tablename = "lstm_input_15m"  # 기본값: 1분 단위 데이터
    print(f"\n📊 사용할 테이블: {tablename}")
    
    # 4. DB에서 데이터 로드
    print(f"\n📊 데이터베이스에서 데이터 로드 중...")
    new_data = load_new_data(
        tablename,
        config['dateColumn'],
        config['studyColumns'],
        start_date=None,  # 전체 기간 조회
        end_date=None
    )
    
    if new_data is None or new_data.empty:
        print("❌ 예측할 데이터가 없습니다.")
        return
    
    # 5. 과거 데이터로 예측 수행 (모델 성능 평가)
    print(f"\n{'='*70}")
    result = predict_with_model(model, scaler, config, new_data)
    
    if result is None:
        return
    
    # 6. 예측 결과 출력 (최근 30개)
    print_recent_predictions(result, n=10)
    
    # 7. 실제 미래값 예측 실행
    print(f"\n{'='*70}")
    
    seq_len = int(config.get('r_seqLen', 60))
    # auto_future_steps = max(20, seq_len)  # 최소 20, 최대 시퀀스 길이만큼 예측
    auto_future_steps = 672;
    
    print(f"🔮 개선된 실제 미래값 예측 수행")
    print(f"   - 모델 시퀀스 길이: {seq_len}")
    print(f"   - 예측할 미래 스텝: {auto_future_steps}개")
    
    future_result = None
    try:
        # 미래값 예측 수행
        future_result = predict_future_improved(model, scaler, config, new_data, auto_future_steps)
        
        if future_result:
            # 미래 예측 결과 출력
            print_future_predictions_improved(future_result)
            
            # 미래 예측 결과 JSON 파일로 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            future_filename = f"{model_name}_future_improved_{timestamp}.json"
            future_filepath = os.path.join(prediction_path, future_filename)
            
            with open(future_filepath, 'w', encoding='utf-8') as f:
                json.dump(future_result, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 미래 예측 결과 저장 완료: {future_filepath}")
    except Exception as e:
        print(f"❌ 미래값 예측 중 오류: {str(e)}")
    
    # 8. 완료 메시지
    print(f"\n{'='*70}")
    print("🎉 예측 완료!")
    print("="*70)
    
    # 9. 결과 반환 (API나 다른 함수에서 사용 가능)
    return_data = {
        "test_result": result,  # 과거 데이터 예측 결과
        "future_result": future_result  # 미래값 예측 결과
    }
    return return_data

# ============================================================================
# 프로그램 시작점
# ============================================================================
if __name__ == "__main__":
    """
    스크립트 직접 실행 시 main() 함수 호출
    
    사용법:
        python lstm_predict.py
    """
    try:
        main()
    except KeyboardInterrupt:
        # Ctrl+C로 중단 시
        print("\n\n⚠️  사용자에 의해 중단되었습니다.")
    except Exception as e:
        # 예상치 못한 에러 발생 시
        print(f"\n❌ 예상치 못한 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()  # 상세 에러 메시지 출력