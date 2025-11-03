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
# EPS: Very small energy outputs를 무시하기 위한 임계값 (kWh 단위 예시)
# 현재는 데이터에 임계값을 주지 않고 학습
# 임계값을 주고싶은경우 PREDICTION_EPS_THRESHOLD 값을 조절
PREDICTION_EPS_THRESHOLD = 0

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
    ※ 현재는 임계값을 0으로 설정 상태

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
    reliable_mask = predictions >= eps_threshold
    unreliable_mask = predictions < eps_threshold
    
    # 인덱스 저장
    reliable_indices = np.where(reliable_mask)[0].tolist()
    unreliable_indices = np.where(unreliable_mask)[0].tolist()
    
    # 개수 계산
    reliable_count = len(reliable_indices)
    unreliable_count = len(unreliable_indices)
    total_count = len(predictions)
    
    # 비율 계산
    reliability_ratio = reliable_count / total_count if total_count > 0 else 0
    
    # 통계 계산
    reliable_stats = None
    if reliable_count > 0:
        reliable_values = predictions[reliable_mask]
        reliable_stats = {
            "min": float(np.min(reliable_values)),
            "max": float(np.max(reliable_values)),
            "mean": float(np.mean(reliable_values)),
            "median": float(np.median(reliable_values)),
            "std": float(np.std(reliable_values))
        }
    
    unreliable_stats = None
    if unreliable_count > 0:
        unreliable_values = predictions[unreliable_mask]
        unreliable_stats = {
            "min": float(np.min(unreliable_values)),
            "max": float(np.max(unreliable_values)),
            "mean": float(np.mean(unreliable_values)),
            "median": float(np.median(unreliable_values))
        }
    
    return {
        "eps_threshold": eps_threshold,
        "total_predictions": int(total_count),
        "reliable_predictions": int(reliable_count),
        "unreliable_predictions": int(unreliable_count),
        "reliability_ratio": float(reliability_ratio),
        "reliable_indices": reliable_indices,
        "unreliable_indices": unreliable_indices,
        "reliable_statistics": reliable_stats,
        "unreliable_statistics": unreliable_stats
    }

# -----------------------------------------------------------------------------
# 미래 예측 수행 함수 (EPS 기반)
# -----------------------------------------------------------------------------
def predict_future_with_eps(model, scaler, config, data, future_steps=96, 
                            eps_threshold=PREDICTION_EPS_THRESHOLD, apply_filter=True):
    """
    EPS 기반 필터링을 적용한 미래값 예측

    동작:
        1. 데이터 전처리 (시간 컬럼 제거, 스케일링)
        2. rolling window 방식으로 future_steps만큼 반복 예측
        3. analyze_prediction_reliability로 EPS 분석 수행
        4. 예측 결과를 JSON 형식으로 포맷하여 반환

    파라미터:
        model: 학습된 LSTM 모델
        scaler: MinMaxScaler 또는 StandardScaler
        config: 모델 설정 dict (targetColumn, dateColumn, r_seqLen 등)
        data: 입력 DataFrame (DB에서 로드한 것)
        future_steps: 예측할 미래 스텝 수
        eps_threshold: EPS 임계값
        apply_filter: 신뢰도 필터 적용 여부

    반환값:
        dict {
            "metadata": {...},
            "reliability_analysis": {...},
            "predictions": [{"date": ..., "predicted_value": ..., "is_reliable": ...}, ...],
            "statistics": {...}
        } 또는 None (오류 발생 시)
    """
    try:
        print(f"\n🔮 미래값 예측 시작")
        print(f"   - 예측 스텝 수: {future_steps}")
        print(f"   - 시퀀스 길이: {config['r_seqLen']}")
        print(f"   - 타겟 컬럼: {config['targetColumn']}")
        print(f"   - EPS 임계값: {eps_threshold}")
        print(f"   - 필터링 적용: {'예' if apply_filter else '아니오'}")
        
        # 시간 컬럼 제거
        dateColumn = config['dateColumn']
        feature_columns = [col for col in data.columns if col != dateColumn]
        
        # 타겟 컬럼 인덱스 확인
        target_col = config['targetColumn']
        if target_col not in feature_columns:
            print(f"❌ 타겟 컬럼 '{target_col}'을 찾을 수 없습니다.")
            return None
        
        target_idx = feature_columns.index(target_col)
        
        # 입력 데이터 준비 (스케일링)
        feature_data = data[feature_columns]
        scaled_data = scaler.transform(feature_data)
        
        # 초기 시퀀스
        seq_len = config['r_seqLen']
        if len(scaled_data) < seq_len:
            print(f"❌ 데이터가 시퀀스 길이({seq_len})보다 짧습니다.")
            return None
        
        # 현재 시퀀스(rolling window)
        current_sequence = scaled_data[-seq_len:].copy()
        
        # 미래 예측을 담을 리스트
        future_predictions = []
        
        # 마지막 날짜 파싱(시작점)
        last_date = pd.to_datetime(data[dateColumn].iloc[-1])
        time_interval = timedelta(minutes=15)  # 15분 간격 (LSTM 데이터에 맞춤)
        
        # rolling window로 미래값 예측 반복
        print(f"   🔄 예측 진행 중...")
        for step in range(future_steps):
            # 모델 입력 형태: [1, seq_len, features]
            X = current_sequence.reshape(1, seq_len, -1)
            
            # 한 스텝 예측
            pred_scaled = model.predict(X, verbose=0)
            pred_value = pred_scaled[0, 0]
            
            # 역스케일링 (타겟 컬럼만)
            # 스케일러가 다변량이면 같은 길이의 dummy 배열 만들어서 역변환
            dummy = np.zeros((1, len(feature_columns)))
            dummy[0, target_idx] = pred_value
            pred_original = scaler.inverse_transform(dummy)[0, target_idx]
            
            future_predictions.append(pred_original)
            
            # 다음 시퀀스 업데이트
            # 현재 시퀀스에서 가장 오래된 행 제거, 새로운 예측을 끝에 추가
            new_row = current_sequence[-1].copy()
            new_row[target_idx] = pred_value
            
            current_sequence = np.vstack([current_sequence[1:], new_row])
            
            # 진행 로그
            if (step + 1) % 100 == 0:
                print(f"      ⏳ {step+1}/{future_steps} 완료")
        
        print(f"   ✅ 예측 완료!")
        
        # EPS 기반 신뢰도 분석
        print(f"\n📊 예측 신뢰도 분석 중...")
        reliability = analyze_prediction_reliability(future_predictions, eps_threshold)
        
        print(f"   - 전체 예측: {reliability['total_predictions']}건")
        print(f"   - 신뢰 가능: {reliability['reliable_predictions']}건")
        print(f"   - 신뢰 불가: {reliability['unreliable_predictions']}건")
        print(f"   - 신뢰율: {reliability['reliability_ratio']*100:.2f}%")
        
        # 예측 결과를 JSON 형식으로 포맷
        predictions_list = []
        for i, pred_val in enumerate(future_predictions):
            future_date = last_date + time_interval * (i + 1)
            is_reliable = i in reliability['reliable_indices']
            
            predictions_list.append({
                "date": convert_to_serializable(future_date),
                "predicted_value": convert_to_serializable(pred_val),
                "is_reliable": is_reliable
            })
        
        # 최종 결과 딕셔너리
        future_result = {
            "metadata": {
                "model_name": config.get('model_name', 'unknown'),
                "target_column": target_col,
                "sequence_length": seq_len,
                "prediction_steps": future_steps,
                "eps_threshold": eps_threshold,
                "filter_applied": apply_filter,
                "last_known_date": convert_to_serializable(last_date),
                "first_prediction_date": convert_to_serializable(last_date + time_interval),
                "last_prediction_date": convert_to_serializable(last_date + time_interval * future_steps)
            },
            "reliability_analysis": reliability,
            "predictions": predictions_list,
            "statistics": {
                "min_predicted": convert_to_serializable(np.min(future_predictions)),
                "max_predicted": convert_to_serializable(np.max(future_predictions)),
                "mean_predicted": convert_to_serializable(np.mean(future_predictions)),
                "median_predicted": convert_to_serializable(np.median(future_predictions)),
                "std_predicted": convert_to_serializable(np.std(future_predictions))
            }
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
def save_predictions_to_db_with_eps(prediction_result, target_table="usage_generation_forecast", 
                                    only_reliable=False):
    """
    미래 예측 결과를 PostgreSQL DB에 저장 (EPS 필터링 옵션)

    파라미터:
        prediction_result: predict_future_with_eps의 반환 dict
        target_table: 저장 대상 테이블명 (carbontwin.<target_table> 사용)
        only_reliable: True이면 is_reliable == True인 예측만 저장

    동작:
        - 기존 동일 time_point 레코드는 DELETE로 제거(중복 방지)
        - INSERT로 새 레코드 추가 (time_point, forecast_usage_kwh, reg_dt)
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
                    
                    # 삽입: forecast_usage_kwh 컬럼명으로 변경
                    insert_query = text(f"""
                    INSERT INTO carbontwin.{target_table} 
                        (time_point, forecast_usage_kwh, reg_dt)
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
        python lstm_model_load.py

    실행시 제공되는 옵션:
        - 사용자가 콘솔에서 모드를 선택하고 EPS 값 입력 가능
        - 기본적으로 모델명과 테이블명은 스크립트 내부의 기본값을 사용
    """
    try:
        model_name = "solar-hybrid-seq-2-test-20251017-test-no-add-usage_kwh"
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