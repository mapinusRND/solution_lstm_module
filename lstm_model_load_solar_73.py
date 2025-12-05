# -*- coding: utf-8 -*-
"""
Title   : EPS 임계값 필터링이 적용된 LSTM 예측 스크립트
Author  : 주성중 / (주)맵인어스
Description: 
    - 학습된 LSTM 모델로 신규 데이터 예측 수행
    - 최근 1일치 데이터로 미래 7일 예측
    - EPS 임계값 기반 예측 신뢰도 필터링 추가
    - PostgreSQL DB 저장 기능
Version : 2.7
Date    : 2025-11-05
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

root = "D:/work/lstm"
model_path = os.path.abspath(root + "/saved_models")
cust_id = "73";

# -----------------------------------------------------------------------------
# 🔥 EPS 임계값 설정 (전역 변수)
# -----------------------------------------------------------------------------
PREDICTION_EPS_THRESHOLD = 0

# -----------------------------------------------------------------------------
# DB 연결 함수
# -----------------------------------------------------------------------------
def get_db_engine():
    """PostgreSQL 데이터베이스 연결 엔진 생성"""
    # connection_string = "postgresql://postgres:mapinus%401004@10.10.10.201:5434/postgres"
    connection_string = "postgresql://postgres:mapinus@10.10.10.201:5432/postgres"
    return create_engine(connection_string)

def convert_to_serializable(obj):
    """NumPy 및 Pandas의 특수 타입을 JSON 직렬화 가능한 Python 기본 타입으로 변환"""
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
def load_new_data(tablename, dateColumn, studyColumns, start_date=None, end_date=None, days_limit=1):
    """PostgreSQL DB에서 예측할 신규 데이터를 로드 (최근 1일치)"""
    try:
        engine = get_db_engine()
        
        print("cust_id : ",cust_id);
        if start_date is None and end_date is None:
            query = f"""
            SELECT {studyColumns},{dateColumn}
            FROM carbontwin.{tablename}
            WHERE {dateColumn} IS NOT NULL  
              AND time_point >= (
                    SELECT MAX(time_point) - INTERVAL '{days_limit} days'
                    FROM carbontwin.{tablename}
                    WHERE time_point IS NOT null
                )
              AND cust_id = {cust_id}
            ORDER BY {dateColumn} ASC
            """
        else:
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
        
        data = pd.read_sql_query(query, engine)
        print(f"✅ 신규 데이터 로드 완료: {len(data)}행")
        
        if len(data) > 0 and dateColumn in data.columns:
            min_date = pd.to_datetime(data[dateColumn]).min()
            max_date = pd.to_datetime(data[dateColumn]).max()
            print(f"   📅 데이터 기간: {min_date} ~ {max_date}")
        
        return data
        
    except Exception as e:
        print(f"❌ 데이터 로드 오류: {str(e)}")
        return None

# -----------------------------------------------------------------------------
# 모델 로드
# -----------------------------------------------------------------------------
def load_trained_model(model_name):
    """저장된 LSTM 모델, 스케일러, 설정 파일을 로드"""
    try:
        model_file = os.path.join(model_path, f"{model_name}.h5")
        scaler_file = os.path.join(model_path, f"{model_name}_scaler.pkl")
        config_file = os.path.join(model_path, f"{model_name}_config.json")
        
        if not all(os.path.exists(f) for f in [model_file, scaler_file, config_file]):
            print(f"❌ 필요한 파일을 찾을 수 없습니다.")
            return None, None, None
        
        print(f"📂 모델 로드 중: {model_name}")
        
        model = load_model(model_file, compile=False)
        model.compile(optimizer='adam', loss='mse')
        
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
    """예측값의 신뢰도를 EPS 임계값 기반으로 분석"""
    predictions = np.array(predictions)
    
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
    """EPS 임계값 기반으로 필터링된 예측값을 테이블 형식으로 출력"""
    predictions = np.array(predictions)
    
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
            confidence = "높음" if pred_val > eps_threshold * 10 else "보통"
            
            print(f"{idx:>6} {date_str:<25} {pred_val:>12.4f} {confidence:>10}")
        
        if len(reliable_indices) > 20:
            print(f"... ({len(reliable_indices) - 20}개 더 있음)")
        
        print(f"{'='*90}")
    else:
        print(f"\n⚠️  신뢰 가능한 예측값이 없습니다!")
        print(f"   💡 모델 재학습을 권장합니다.")
    
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
    """EPS 임계값 필터링이 적용된 미래값 예측 (1일 데이터로 7일 예측)"""
    try:
        dateColumn = config['dateColumn']
        studyColumns = config['studyColumns']
        targetColumn = config['targetColumn']
        seq_len = int(config['r_seqLen'])
        pred_days = int(config['r_predDays'])
        
        if future_steps is None:
            future_steps = 672  # 7일 = 7 * 96 (15분 간격)
        
        study_columns_list = [col.strip() for col in studyColumns.split(',')]
        target_idx = study_columns_list.index(targetColumn)
        
        if dateColumn in new_data.columns:
            last_date = pd.to_datetime(new_data[dateColumn].iloc[-1])
        else:
            last_date = datetime.now()
        
        print(f"\n🔮 EPS 필터링 미래값 예측 시작...")
        print(f"   - 입력 데이터: {len(new_data)}개 (최근 1일치)")
        print(f"   - 시퀀스 길이: {seq_len}개")
        print(f"   - 예측 스텝: {future_steps}개 (7일치)")
        print(f"   - EPS 임계값: {eps_threshold}")
        print(f"   - 필터링 적용: {'예' if apply_filter else '아니오'}")
        print(f"   - 마지막 데이터 시간: {last_date}")
        
        data_for_prediction = new_data[study_columns_list].astype(float)
        
        # 🔥 NULL 체크 및 처리
        # 🔥 NULL 체크 및 처리
        null_count = data_for_prediction.isnull().sum().sum()
        if null_count > 0:
            print(f"⚠️  경고: {null_count}개의 NULL 값 발견!")
            print(f"   NULL 값 분포:\n{data_for_prediction.isnull().sum()}")
            
            print(f"\n   📝 1단계: ffill/bfill 적용...")
            data_for_prediction = data_for_prediction.ffill().bfill()
            
            remaining_nulls = data_for_prediction.isnull().sum().sum()
            
            if remaining_nulls > 0:
                print(f"   📝 2단계: 남은 {remaining_nulls}개 NULL 처리 중...")
                
                # 🔥 현재 월 확인
                current_month = datetime.now().month
                
                # 🔥 월별 기본값 (한국 평균 기온/습도)
                MONTHLY_DEFAULTS = {
                    1:  {'temp': 0, 'humi': 55},    # 1월
                    2:  {'temp': 3, 'humi': 55},    # 2월
                    3:  {'temp': 8, 'humi': 60},    # 3월
                    4:  {'temp': 14, 'humi': 60},   # 4월
                    5:  {'temp': 19, 'humi': 65},   # 5월
                    6:  {'temp': 23, 'humi': 70},   # 6월
                    7:  {'temp': 26, 'humi': 80},   # 7월
                    8:  {'temp': 27, 'humi': 80},   # 8월
                    9:  {'temp': 22, 'humi': 75},   # 9월
                    10: {'temp': 16, 'humi': 70},  # 10월
                    11: {'temp': 9, 'humi': 65},   # 11월
                    12: {'temp': 2, 'humi': 60},   # 12월
                }
                
                print(f"      - 현재 월: {current_month}월")
                print(f"      - 기본 온도: {MONTHLY_DEFAULTS[current_month]['temp']}°C")
                print(f"      - 기본 습도: {MONTHLY_DEFAULTS[current_month]['humi']}%")
                
                # 모든 컬럼에 대해 처리
                for col in data_for_prediction.columns:
                    null_count_col = data_for_prediction[col].isnull().sum()
                    
                    if null_count_col > 0:
                        # 평균값 계산
                        mean_val = data_for_prediction[col].mean()
                        
                        # 평균값도 NaN이면 (모든 값이 NULL) 컬럼별 기본값 사용
                        if pd.isna(mean_val):
                            # 온도/습도 컬럼은 월별 기본값 사용
                            if 'temp' in col.lower():
                                fill_value = MONTHLY_DEFAULTS[current_month]['temp']
                                print(f"      - {col}: {fill_value}°C로 채움 (월별 평균 온도)")
                            elif 'humi' in col.lower():
                                fill_value = MONTHLY_DEFAULTS[current_month]['humi']
                                print(f"      - {col}: {fill_value}%로 채움 (월별 평균 습도)")
                            else:
                                # 다른 컬럼은 0
                                fill_value = 0
                                print(f"      - {col}: 0으로 채움 (기본값)")
                        else:
                            fill_value = mean_val
                            print(f"      - {col}: {fill_value:.2f}로 채움 (데이터 평균값)")
                        
                        data_for_prediction[col] = data_for_prediction[col].fillna(fill_value)
            
            # 최종 확인
            final_nulls = data_for_prediction.isnull().sum().sum()
            if final_nulls > 0:
                print(f"\n❌ 오류: {final_nulls}개의 NULL을 처리할 수 없습니다!")
                return None
            else:
                print(f"\n   ✅ 모든 NULL 값 처리 완료")
        
        if len(data_for_prediction) < seq_len:
            raise ValueError(f"데이터 부족: {len(data_for_prediction)}개 (최소 {seq_len}개 필요)")
        
        data_scaled = scaler.transform(data_for_prediction)
        
        # 🔥 시간 간격 계산 개선 (가장 중요!)
        time_delta = pd.Timedelta(minutes=15)  # 기본값 먼저 설정
        
        if dateColumn in new_data.columns and len(new_data) > 1:
            dates = pd.to_datetime(new_data[dateColumn])
            
            # 여러 구간의 시간 간격 계산
            time_diffs = dates.diff().dropna()
            
            if len(time_diffs) > 0:
                # 가장 빈번한 간격 (mode)
                mode_diff = time_diffs.mode()
                if not mode_diff.empty and mode_diff[0] > pd.Timedelta(0):
                    time_delta = mode_diff[0]
                    print(f"   ⏰ 계산된 시간 간격: {time_delta} ({time_delta.total_seconds()/60:.0f}분)")
                else:
                    # mode가 없으면 median
                    median_diff = time_diffs.median()
                    if median_diff > pd.Timedelta(0) and median_diff <= pd.Timedelta(hours=1):
                        time_delta = median_diff
                        print(f"   ⏰ 계산된 시간 간격(중앙값): {time_delta} ({time_delta.total_seconds()/60:.0f}분)")
                    else:
                        print(f"   ⚠️ 비정상적인 간격 감지, 기본값 15분 사용")
            else:
                print(f"   ⏰ 기본 시간 간격 사용: 15분")
        else:
            print(f"   ⏰ 기본 시간 간격 사용: 15분")
        
        # 🔥 디버그 출력
        print(f"\n   🔍 시간 설정 확인:")
        print(f"      - last_date = {last_date}")
        print(f"      - time_delta = {time_delta}")
        print(f"      - 테스트: 다음 시간 = {last_date + time_delta}")
        
        # 🔥 안전장치: time_delta가 0 이하면 강제로 15분 설정
        if time_delta <= pd.Timedelta(0):
            print(f"   ⚠️ 경고: time_delta가 0 또는 음수! 강제로 15분 설정")
            time_delta = pd.Timedelta(minutes=15)
        
        current_sequence = data_scaled[-seq_len:].copy()
        
        future_predictions = []
        future_predictions_raw = []
        future_dates = []
        prediction_confidence = []
        
        recent_data = data_for_prediction[targetColumn].tail(100)
        recent_positive = recent_data[recent_data > eps_threshold]
        baseline = recent_positive.median() if len(recent_positive) > 0 else eps_threshold
        
        print(f"   📊 예측 기준값: {baseline:.4f}")
        print(f"\n   🚀 예측 시작...\n")
        
        # 🔥 예측 루프
        for step in range(future_steps):
            # 명확한 날짜 계산
            next_date = last_date + (time_delta * (step + 1))
            hour = next_date.hour
            
            # 처음 10개와 마지막 5개만 출력
            if step < 10:
                print(f"   Step {step:3d}: {next_date.strftime('%Y-%m-%d %H:%M:%S')}")
            elif step == 10:
                print(f"   ... ({future_steps - 15}개 생략)")
            elif step >= future_steps - 5:
                print(f"   Step {step:3d}: {next_date.strftime('%Y-%m-%d %H:%M:%S')}")
            
            input_data = current_sequence.reshape(1, seq_len, len(study_columns_list))
            pred_scaled = model.predict(input_data, verbose=0)[0, 0]
            
            pred_original = pred_scaled * scaler.scale_[target_idx] + scaler.mean_[target_idx]
            
            future_predictions_raw.append(pred_original)
            
            if apply_filter:
                if pred_original <= eps_threshold:
                    pred_filtered = 0.0
                else:
                    if 6 <= hour < 18:
                        pred_filtered = pred_original
                    else:
                        pred_filtered = max(0, pred_original * 0.1)
            else:
                pred_filtered = max(0, pred_original)
            
            if pred_filtered > eps_threshold:
                confidence = min(1.0, pred_filtered / (baseline * 2))
            else:
                confidence = 0.0
            
            future_predictions.append(pred_filtered)
            future_dates.append(next_date)
            prediction_confidence.append(confidence)
            
            new_point = current_sequence[-1].copy()
            new_point_scaled = (pred_filtered - scaler.mean_[target_idx]) / scaler.scale_[target_idx]
            new_point[target_idx] = new_point_scaled
            
            current_sequence = np.vstack([current_sequence[1:], new_point])
            
            if (step + 1) % 100 == 0:
                print(f"   ⏳ 진행: {step + 1}/{future_steps} 스텝 완료")
        
        print(f"\n✅ 예측 완료!")
        
        # 날짜 범위 확인
        if len(future_dates) > 0:
            print(f"   📅 예측 기간: {future_dates[0].strftime('%Y-%m-%d %H:%M:%S')} ~ {future_dates[-1].strftime('%Y-%m-%d %H:%M:%S')}")
        
        reliability = analyze_prediction_reliability(future_predictions, eps_threshold)
        
        print(f"\n📊 예측 결과 요약:")
        print(f"   - 전체 예측: {len(future_predictions)}개")
        print(f"   - 신뢰 가능: {reliability['reliable_predictions']}개 "
              f"({reliability['reliability_ratio']*100:.1f}%)")
        print(f"   - 신뢰 불가: {reliability['unreliable_predictions']}개")
        print(f"   - 예측값 범위: {min(future_predictions):.4f} ~ {max(future_predictions):.4f}")
        
        print_predictions_with_eps_filter(future_predictions, future_dates, eps_threshold)
        
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
            })
        
        future_result["statistics"] = {
            "min_predicted": convert_to_serializable(np.min(future_predictions)),
            "max_predicted": convert_to_serializable(np.max(future_predictions)),
            "mean_predicted": convert_to_serializable(np.mean(future_predictions)),
            "median_predicted": convert_to_serializable(np.median(future_predictions)),
            "std_predicted": convert_to_serializable(np.std(future_predictions))
        }
        
        return future_result
        
    except Exception as e:
        print(f"❌ 미래값 예측 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# -----------------------------------------------------------------------------
# 🔥 EPS 필터링 적용한 DB 저장 함수
# -----------------------------------------------------------------------------
def save_predictions_to_db_with_eps(prediction_result,target_table="solar_generation_forecast", 
                                    only_reliable=False):
    """미래 예측 결과를 PostgreSQL DB에 저장"""
    if prediction_result is None:
        print("❌ 저장할 예측 결과가 없습니다.")
        return 0, 0
    
    try:
        engine = get_db_engine()
        predictions = prediction_result.get('predictions', [])
        
        if not predictions:
            print("❌ 예측 데이터가 비어있습니다.")
            return 0, 0
        
        if only_reliable:
            predictions = [p for p in predictions if p.get('is_reliable', False)]
            print(f"\n📊 신뢰 가능한 예측만 저장: {len(predictions)}건")
        
        print(f"\n💾 예측 결과 DB 저장 시작...")
        print(f"   - 대상 테이블: carbontwin.{target_table}")
        print(f"   - 저장할 데이터: {len(predictions)}건")
        
        success_count = 0
        fail_count = 0
        
        # 🔥 with문 사용 + 명시적 트랜잭션 관리
        with engine.begin() as connection:
            # 🔥 타임존 설정
            connection.execute(text("SET timezone = 'Asia/Seoul'"))
            
            try:
                for pred in predictions:
                    time_point = pred['date']
                    forecast_value = pred['predicted_value']
                    
                    # 기존 데이터 삭제
                    delete_query = text(f"""
                    DELETE FROM carbontwin.{target_table}
                    WHERE time_point = :time_point
                    """)
                    
                    connection.execute(delete_query, {"time_point": time_point})
                    
                    # 새 데이터 삽입
                    insert_query = text(f"""
                    INSERT INTO carbontwin.{target_table} 
                        (time_point, forecast_solar_kwh, reg_dt, cust_id)
                    VALUES 
                        (:time_point, :forecast_value, NOW(), {cust_id}) 
                    """) 
                    
                    connection.execute(
                        insert_query,
                        {
                            "time_point": time_point,
                            "forecast_value": forecast_value
                        }
                    )
                    success_count += 1
                    
                    if success_count % 100 == 0:
                        print(f"   ⏳ 진행: {success_count}/{len(predictions)} 건")
                
                # 🔥 with문이 자동으로 커밋함
                print(f"✅ DB 저장 완료!")
                print(f"   - 성공: {success_count}건")
                
            except Exception as e:
                # 🔥 with문이 자동으로 롤백함
                print(f"❌ DB 저장 중 오류 (자동 롤백됨): {str(e)}")
                import traceback
                traceback.print_exc()
                fail_count = len(predictions) - success_count
        
        return success_count, fail_count
        
    except Exception as e:
        print(f"❌ DB 연결 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0, len(predictions) if predictions else 0

# -----------------------------------------------------------------------------
# 메인 실행 함수
# -----------------------------------------------------------------------------
def main(model_name=None, tablename=None, save_to_db=True, only_reliable=False, 
         eps_threshold=PREDICTION_EPS_THRESHOLD, apply_filter=True):
    """메인 실행 함수 - 최근 1일 데이터로 미래 7일 예측"""
    print("=" * 70)
    print("🔮 LSTM 미래 예측 시스템 (1일 → 7일)")
    print("=" * 70)
    
    model, scaler, config = load_trained_model(model_name)
    
    if model is None:
        return None
    
    print(f"\n📊 데이터베이스에서 데이터 로드 중...")
    new_data = load_new_data(tablename, config['dateColumn'], config['studyColumns'], days_limit=1)
    
    if new_data is None or new_data.empty:
        print("❌ 예측할 데이터가 없습니다.")
        return None
    
    future_steps = 672  # 7일
    
    print(f"\n🔮 미래값 예측 수행")
    print(f"   - 입력: 최근 1일 ({len(new_data)}개 데이터)")
    print(f"   - 출력: 미래 7일 ({future_steps}개 예측)")
    print(f"   - EPS 임계값: {eps_threshold}")
    print(f"   - 필터링 적용: {'예' if apply_filter else '아니오'}")
    
    future_result = predict_future_with_eps(
        model, scaler, config, new_data, future_steps,
        eps_threshold, apply_filter
    )
    
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
    """최근 1일 데이터로 미래 7일 예측"""
    try:
        #model_name = "solar-hybrid-seq-2-test-20251017-test-no-add-test"
        model_name = "usage_kwh_model_test_solar"
        tablename = "lstm_input_15m"
        
        print("\n" + "=" * 80)
        print("🔍 실행 모드: 1일 데이터 → 7일 예측")
        print("=" * 80)
        
        eps_threshold = PREDICTION_EPS_THRESHOLD
        
        print(f"\n⚙️  설정:")
        print(f"   - EPS 임계값: {eps_threshold}")
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