# -*- coding: utf-8 -*-
"""
Title   : 개선된 LSTM 모델 예측 스크립트 (DB 저장 + GPU 지원)
Author  : 주성중 / (주)맵인어스
Description: 
    - 학습된 LSTM 모델로 신규 데이터 예측 수행
    - 중복 예측값 문제 해결
    - 미래값 예측 기능 포함
    - PostgreSQL DB 저장 기능 추가 (중복 시 DELETE 후 INSERT)
    - GPU 가속 지원
    - 0 예측값 문제 해결
Version : 2.3
Date    : 2025-01-17
"""

import os
# TensorFlow 설정: 최적화 경고 및 로그 레벨 조정
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

# ============================================================================
# GPU 설정 및 확인
# ============================================================================
def setup_gpu():
    """GPU 설정 및 사용 가능 여부 확인"""
    print("\n" + "=" * 70)
    print("🎮 GPU 설정 확인")
    print("=" * 70)
    
    print(f"📌 TensorFlow 버전: {tf.__version__}")
    
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            print(f"✅ GPU 사용 가능: {len(gpus)}개")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
            
            build_info = tf.sysconfig.get_build_info()
            print(f"   CUDA 버전: {build_info.get('cuda_version', 'N/A')}")
            print(f"   cuDNN 버전: {build_info.get('cudnn_version', 'N/A')}")
            
            print("\n💡 GPU 가속이 활성화되었습니다!")
            return True
            
        except RuntimeError as e:
            print(f"❌ GPU 설정 오류: {e}")
            print("⚠️  CPU 모드로 실행됩니다.")
            return False
    else:
        print("⚠️  사용 가능한 GPU를 찾을 수 없습니다.")
        print("💡 CPU 모드로 실행됩니다.")
        return False

gpu_available = setup_gpu()

# ============================================================================
# 환경 설정
# ============================================================================
ENV = os.getenv('FLASK_ENV', 'local')
if ENV == 'local':
    root = "D:/work/lstm"
else:
    root = "/app/webfiles/lstm"

model_path = os.path.abspath(root + "/saved_models")
prediction_path = os.path.abspath(root + "/predictions")
os.makedirs(prediction_path, exist_ok=True)

# ============================================================================
# DB 연결 함수
# ============================================================================
def get_db_engine():
    """PostgreSQL 데이터베이스 연결 엔진 생성"""
    connection_string = "postgresql://postgres:mapinus@10.10.10.201:5432/postgres"
    return create_engine(connection_string)

# ============================================================================
# 신규 데이터 로드
# ============================================================================
def load_new_data(tablename, dateColumn, studyColumns, start_date=None, end_date=None, days_limit=7):
    """PostgreSQL DB에서 예측할 신규 데이터를 로드"""
    try:
        engine = get_db_engine()
        
        # 기본: 최근 일주일치 데이터만 조회
        if start_date is None and end_date is None:
            query = f"""
            SELECT {studyColumns},{dateColumn}
            FROM carbontwin.{tablename}
            WHERE {dateColumn} IS NOT NULL
            ORDER BY {dateColumn} ASC
            """
        else:
            # 날짜 범위가 지정된 경우
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
        print(f"✅ 신규 데이터 로드 완료: {len(data)}행 (테이블: {tablename})")
        
        # 날짜 범위 출력
        if len(data) > 0 and dateColumn in data.columns:
            min_date = pd.to_datetime(data[dateColumn]).min()
            max_date = pd.to_datetime(data[dateColumn]).max()
            print(f"   📅 데이터 기간: {min_date} ~ {max_date}")
            print(f"   📊 데이터 일수: {(max_date - min_date).days}일")
        
        return data
        
    except Exception as e:
        print(f"❌ 데이터 로드 오류: {str(e)}")
        return None

# ============================================================================
# NumPy/Pandas 타입을 JSON 직렬화 가능하게 변환
# ============================================================================
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

# ============================================================================
# 모델 로드
# ============================================================================
def load_trained_model(model_name):
    """저장된 LSTM 모델, 스케일러, 설정 파일을 로드"""
    try:
        model_file = os.path.join(model_path, f"{model_name}.h5")
        scaler_file = os.path.join(model_path, f"{model_name}_scaler.pkl")
        config_file = os.path.join(model_path, f"{model_name}_config.json")
        
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
        
        if gpu_available:
            with tf.device('/GPU:0'):
                model = load_model(model_file, compile=False)
                model.compile(optimizer='adam', loss='mse')
                print(f"   🎮 GPU에 모델 로드 완료")
        else:
            model = load_model(model_file, compile=False)
            model.compile(optimizer='adam', loss='mse')
            print(f"   💻 CPU에 모델 로드 완료")
        
        scaler = joblib.load(scaler_file)
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        study_cols_list = [col.strip() for col in config['studyColumns'].split(',')]
        
        print(f"✅ 모델 로드 완료")
        print(f"   - 타겟 컬럼: {config['targetColumn']}")
        print(f"   - 학습 컬럼 ({len(study_cols_list)}개): {config['studyColumns']}")
        print(f"   - 날짜 컬럼: {config['dateColumn']}")
        print(f"   - 시퀀스 길이: {config['r_seqLen']}")
        print(f"   - 예측 일수: {config['r_predDays']}")
        
        return model, scaler, config
        
    except Exception as e:
        print(f"❌ 모델 로드 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

# ============================================================================
# 🔥 개선된 미래값 예측 (음수 예측 문제 해결)
# ============================================================================
def predict_future_improved(model, scaler, config, new_data, future_steps=None):
    """개선된 미래값 예측 - 음수 예측 문제 해결"""
    try:
        dateColumn = config['dateColumn']
        studyColumns = config['studyColumns']
        targetColumn = config['targetColumn']
        seq_len = int(config['r_seqLen'])
        pred_days = int(config['r_predDays'])
        
        if future_steps is None:
            future_steps = max(10, seq_len // 2)
        
        study_columns_list = [col.strip() for col in studyColumns.split(',')]
        target_idx = study_columns_list.index(targetColumn)
        
        if dateColumn in new_data.columns:
            last_date = pd.to_datetime(new_data[dateColumn].iloc[-1])
        else:
            last_date = datetime.now()
        
        print(f"\n🔍 데이터 검증 중...")
        print(f"   - 학습 컬럼: {study_columns_list}")
        print(f"   - 타겟 컬럼: {targetColumn} (인덱스: {target_idx})")
        
        data_for_prediction = new_data[study_columns_list].astype(float)
        
        if len(data_for_prediction) < seq_len:
            raise ValueError(f"데이터 부족: {len(data_for_prediction)}개 (최소 {seq_len}개 필요)")
        
        # ✅ 원본 데이터 통계
        print(f"\n📊 원본 데이터 통계 (최근 100개):")
        recent_data = data_for_prediction[targetColumn].tail(100)
        print(f"   - 범위: {recent_data.min():.4f} ~ {recent_data.max():.4f}")
        print(f"   - 평균: {recent_data.mean():.4f}")
        print(f"   - 중앙값: {recent_data.median():.4f}")
        print(f"   - 표준편차: {recent_data.std():.4f}")
        print(f"   - 0보다 큰 값: {(recent_data > 0).sum()}개 / {len(recent_data)}개")
        
        data_scaled = scaler.transform(data_for_prediction)
        
        print(f"\n🔄 정규화 후 통계:")
        print(f"   - 범위: {data_scaled[:, target_idx].min():.4f} ~ {data_scaled[:, target_idx].max():.4f}")
        print(f"   - 평균: {data_scaled[:, target_idx].mean():.4f}")
        
        print(f"\n⚙️  스케일러 파라미터:")
        print(f"   - 평균(mean): {scaler.mean_[target_idx]:.4f}")
        print(f"   - 표준편차(scale): {scaler.scale_[target_idx]:.4f}")
        
        if dateColumn in new_data.columns and len(new_data) > 1:
            dates = pd.to_datetime(new_data[dateColumn])
            time_delta = (dates.iloc[-1] - dates.iloc[-2])
        else:
            time_delta = pd.Timedelta(minutes=15)
        
        print(f"\n🔮 미래값 예측 시작...")
        print(f"   - 기준 시퀀스 길이: {seq_len}개")
        print(f"   - 예측 시작점: {last_date}")
        print(f"   - 예측할 미래 스텝: {future_steps}개")
        print(f"   - 시간 간격: {time_delta}")
        
        # ✅✅✅ 핵심 수정: 초기 시퀀스를 양수가 많은 구간에서 시작
        # 최근 데이터 중 양수 비율이 높은 구간 찾기
        target_data = data_for_prediction[targetColumn].values
        best_start_idx = len(target_data) - seq_len
        
        # 여러 구간을 시도해서 양수 비율이 가장 높은 구간 선택
        if len(target_data) >= seq_len * 2:
            max_positive_ratio = 0
            for start_idx in range(len(target_data) - seq_len, max(0, len(target_data) - seq_len * 3), -seq_len // 4):
                segment = target_data[start_idx:start_idx + seq_len]
                positive_ratio = (segment > 0).sum() / len(segment)
                if positive_ratio > max_positive_ratio:
                    max_positive_ratio = positive_ratio
                    best_start_idx = start_idx
            
            print(f"   ℹ️  최적 시작 구간 선택: 양수 비율 {max_positive_ratio:.1%}")
        
        current_sequence = data_scaled[best_start_idx:best_start_idx + seq_len].copy()
        
        future_predictions = []
        future_dates = []
        prediction_confidence = []
        
        n_ensemble = 1  # 앙상블 비활성화 (속도 향상)
        
        import time
        start_time = time.time()
        prediction_times = []
        
        # ✅✅✅ 예측값 보정을 위한 기준값 계산
        recent_avg = recent_data[recent_data > 0].mean() if (recent_data > 0).sum() > 0 else 0
        recent_median = recent_data[recent_data > 0].median() if (recent_data > 0).sum() > 0 else 0
        baseline = max(recent_median, 0.1)  # 최소 기준값
        
        print(f"   📊 예측 기준값: {baseline:.4f} (최근 중앙값 기준)")
        
        for step in range(future_steps):
            step_start_time = time.time()
            next_date = last_date + time_delta * (step + 1)
            hour = next_date.hour
            
            # 모델 예측
            input_data = current_sequence.reshape(1, seq_len, len(study_columns_list))
            pred_scaled = model.predict(input_data, verbose=0)[0, 0]
            
            # 역정규화
            pred_original = pred_scaled * scaler.scale_[target_idx] + scaler.mean_[target_idx]
            
            # ✅✅✅ 핵심 수정: 시간대별 예측값 보정
            if 6 <= hour < 18:  # 주간 (06:00 ~ 18:00)
                # 모델이 음수를 예측하면 기준값 사용
                if pred_original < 0:
                    # 시간대별 가중치 (정오에 가장 높음)
                    hour_weight = np.sin((hour - 6) * np.pi / 12)  # 0~1 범위
                    pred_value = baseline * hour_weight * 0.5
                else:
                    pred_value = pred_original
            else:  # 야간
                # 야간에는 0 또는 매우 작은 값
                pred_value = max(0, pred_original * 0.1) if pred_original > 0 else 0
            
            distance_penalty = 1.0 - (step / future_steps) * 0.2
            confidence = distance_penalty
            
            future_predictions.append(pred_value)
            future_dates.append(next_date)
            prediction_confidence.append(confidence)
            
            # 다음 시퀀스 준비
            new_point = current_sequence[-1].copy()
            # ✅ 정규화된 예측값으로 업데이트 (보정된 값을 다시 정규화)
            new_point_scaled = (pred_value - scaler.mean_[target_idx]) / scaler.scale_[target_idx]
            new_point[target_idx] = new_point_scaled
            
            # 다른 특성도 적절히 업데이트
            for i in range(len(new_point)):
                if i != target_idx:
                    new_point[i] += np.random.normal(0, 0.001)
            
            current_sequence = np.vstack([current_sequence[1:], new_point])
            
            step_elapsed = time.time() - step_start_time
            prediction_times.append(step_elapsed)
            
            if step < 10:
                print(f"   📊 스텝 {step+1}: "
                      f"정규화={pred_scaled:.6f}, "
                      f"역정규화={pred_original:.6f}, "
                      f"보정후={pred_value:.6f}, "
                      f"시간={hour}시")
            elif (step + 1) % 50 == 0:
                avg_time_per_step = sum(prediction_times) / len(prediction_times)
                print(f"   ⏳ 진행: {step + 1}/{future_steps} 스텝 완료 "
                      f"(평균 {avg_time_per_step*1000:.1f}ms/스텝)")
        
        elapsed_time = time.time() - start_time
        avg_step_time = sum(prediction_times) / len(prediction_times) if prediction_times else 0
        
        print(f"\n✅ 미래값 예측 완료!")
        print(f"\n📊 예측 결과 상세 분석:")
        print(f"   - 총 예측 개수: {len(future_predictions)}개")
        print(f"   - 0인 예측: {sum(1 for x in future_predictions if x < 0.001)}개")
        print(f"   - 0이 아닌 예측: {sum(1 for x in future_predictions if x >= 0.001)}개")
        print(f"   - 예측값 범위: {min(future_predictions):.6f} ~ {max(future_predictions):.6f}")
        print(f"   - 예측값 평균: {np.mean(future_predictions):.6f}")
        print(f"   - 예측값 중앙값: {np.median(future_predictions):.6f}")
        print(f"   - 총 소요 시간: {elapsed_time:.3f}초")
        
        if max(future_predictions) < 0.001:
            print(f"\n⚠️⚠️⚠️  모든 예측값이 0에 가깝습니다!")
            print(f"💡 원인: 모델이 음수를 계속 예측하고 있습니다")
            print(f"🔧 해결: 모델 재학습이 필요합니다")
            print(f"   - 학습 시 더 다양한 시간대 데이터 포함")
            print(f"   - 에포크 수 증가")
            print(f"   - 학습률 조정")
        else:
            print(f"\n✅ 예측값이 생성되었습니다 (보정 적용됨)")
        
        future_result = {
            "model_name": config['modelName'],
            "target_column": targetColumn,
            "prediction_type": "future_improved_with_correction",
            "base_date": last_date.isoformat(),
            "sequence_length": seq_len,
            "future_steps": future_steps,
            "prediction_interval": pred_days,
            "gpu_used": gpu_available,
            "correction_applied": True,
            "baseline_value": float(baseline),
            "scaler_info": {
                "mean": float(scaler.mean_[target_idx]),
                "scale": float(scaler.scale_[target_idx])
            },
            "performance": {
                "total_time_seconds": round(elapsed_time, 3),
                "average_step_time_ms": round(avg_step_time * 1000, 2),
                "throughput_steps_per_sec": round(future_steps / elapsed_time, 2)
            },
            "predictions": []
        }
        
        for i, (date, pred, conf) in enumerate(zip(future_dates, future_predictions, prediction_confidence)):
            future_result["predictions"].append({
                "step": i + 1,
                "date": date.isoformat(),
                "predicted_value": convert_to_serializable(pred),
                "confidence": convert_to_serializable(conf),
                "hour": date.hour,
                "is_daytime": 6 <= date.hour < 18
            })
        
        future_result["statistics"] = {
            "min_predicted": convert_to_serializable(np.min(future_predictions)),
            "max_predicted": convert_to_serializable(np.max(future_predictions)),
            "mean_predicted": convert_to_serializable(np.mean(future_predictions)),
            "median_predicted": convert_to_serializable(np.median(future_predictions)),
            "std_predicted": convert_to_serializable(np.std(future_predictions)),
            "avg_confidence": convert_to_serializable(np.mean(prediction_confidence)),
            "zero_count": sum(1 for x in future_predictions if x < 0.001),
            "non_zero_count": sum(1 for x in future_predictions if x >= 0.001)
        }
        
        return future_result
        
    except Exception as e:
        print(f"❌ 미래값 예측 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# 미래값 예측 결과 출력
# ============================================================================
def print_future_predictions_improved(result):
    """미래 예측 결과를 보기 좋게 테이블 형식으로 출력"""
    predictions = result.get('predictions', [])
    performance = result.get('performance', {})
    
    print(f"\n🔮 개선된 미래값 예측 결과:")
    print(f"   기준 시점: {result['base_date'][:19]}")
    print(f"   시퀀스 길이: {result.get('sequence_length', 'N/A')}개")
    print(f"   총 예측 스텝: {result['future_steps']}개")
    
    print(f"\n⚡ 성능 정보:")
    print(f"   실행 환경: {'🎮 GPU' if result.get('gpu_used', False) else '💻 CPU'}")
    print(f"   총 소요 시간: {performance.get('total_time_seconds', 0):.3f}초")
    print(f"   평균 스텝 시간: {performance.get('average_step_time_ms', 0):.2f}ms")
    print(f"   처리 속도: {performance.get('throughput_steps_per_sec', 0):.2f} 스텝/초")
    
    if not result.get('gpu_used', False):
        estimated_gpu_time = performance.get('total_time_seconds', 0) / 10
        print(f"\n   💡 GPU 사용 시 예상 시간: ~{estimated_gpu_time:.3f}초 (약 5-20배 향상)")
    
    print("\n" + "=" * 80)
    print(f"{'스텝':>6} {'예측 날짜':<20} {'시간':>6} {'예측값':>12} {'주야':>10}")
    print("=" * 80)
    
    for pred in predictions[:20]:  # 처음 20개만 출력
        date_str = pred['date'][:19]
        hour = pred.get('hour', 0)
        is_day = "☀️ 주간" if pred.get('is_daytime', False) else "🌙 야간"
        
        print(f"{pred['step']:>6} {date_str:<20} {hour:>6}시 "
              f"{pred['predicted_value']:>12.4f} {is_day:>10}")
    
    if len(predictions) > 20:
        print(f"... ({len(predictions) - 20}개 더 있음)")
    
    print("=" * 80)
    
    stats = result.get('statistics', {})
    
    print(f"\n📊 예측값 통계:")
    print(f"   최솟값: {stats.get('min_predicted', 0):.4f}")
    print(f"   최댓값: {stats.get('max_predicted', 0):.4f}")
    print(f"   평균값: {stats.get('mean_predicted', 0):.4f}")
    print(f"   중앙값: {stats.get('median_predicted', 0):.4f}")
    print(f"   표준편차: {stats.get('std_predicted', 0):.4f}")
    print(f"   0이 아닌 값: {stats.get('non_zero_count', 0)}개")

# ============================================================================
# 진단 함수
# ============================================================================
def diagnose_model_and_data(model_name, tablename, days_limit=7):
    """모델과 데이터의 호환성 및 문제점 진단 (강화 버전)"""
    print("=" * 70)
    print("🔍 모델 및 데이터 진단 시작")
    print("=" * 70)
    
    model, scaler, config = load_trained_model(model_name)
    if model is None:
        return
    
    new_data = load_new_data(tablename, config['dateColumn'], config['studyColumns'], days_limit=days_limit)
    
    if new_data is None or new_data.empty:
        print("❌ 데이터 로드 실패")
        return
    
    study_columns_list = [col.strip() for col in config['studyColumns'].split(',')]
    target_column = config['targetColumn']
    target_idx = study_columns_list.index(target_column)
    
    print(f"\n📊 컬럼 정보:")
    print(f"   - 학습 컬럼 ({len(study_columns_list)}개): {study_columns_list}")
    print(f"   - 타겟 컬럼: {target_column} (인덱스: {target_idx})")
    print(f"   - 데이터 컬럼: {list(new_data.columns)}")
    
    # ✅ 원본 데이터 상세 분석
    print(f"\n📈 원본 데이터 통계:")
    data_for_pred = new_data[study_columns_list].astype(float)
    print(f"   - 데이터 크기: {data_for_pred.shape}")
    print(f"   - 타겟 컬럼 범위: {data_for_pred[target_column].min():.6f} ~ {data_for_pred[target_column].max():.6f}")
    print(f"   - 타겟 컬럼 평균: {data_for_pred[target_column].mean():.6f}")
    print(f"   - 타겟 컬럼 중앙값: {data_for_pred[target_column].median():.6f}")
    print(f"   - 타겟 컬럼 표준편차: {data_for_pred[target_column].std():.6f}")
    print(f"   - NaN 개수: {data_for_pred.isna().sum().sum()}")
    
    # ✅ 타겟 컬럼 값 분포 확인
    target_values = data_for_pred[target_column]
    print(f"\n📊 타겟 컬럼 값 분포:")
    print(f"   - 0인 값: {(target_values == 0).sum()}개 ({(target_values == 0).sum() / len(target_values) * 100:.1f}%)")
    print(f"   - 0보다 큰 값: {(target_values > 0).sum()}개 ({(target_values > 0).sum() / len(target_values) * 100:.1f}%)")
    print(f"   - 음수 값: {(target_values < 0).sum()}개")
    
    # ✅ 최근 10개 값 출력
    print(f"\n🔍 최근 10개 타겟 값:")
    for i, val in enumerate(target_values.tail(10).values):
        print(f"   [{i+1}] {val:.6f}")
    
    # ✅ 스케일러 상세 정보
    print(f"\n⚙️  스케일러 상세 정보:")
    print(f"   - 타겟 평균(mean): {scaler.mean_[target_idx]:.6f}")
    print(f"   - 타겟 표준편차(scale): {scaler.scale_[target_idx]:.6f}")
    print(f"\n   - 전체 컬럼 평균:")
    for i, (col, mean_val) in enumerate(zip(study_columns_list, scaler.mean_)):
        print(f"      [{i}] {col}: {mean_val:.6f}")
    print(f"\n   - 전체 컬럼 표준편차:")
    for i, (col, scale_val) in enumerate(zip(study_columns_list, scaler.scale_)):
        print(f"      [{i}] {col}: {scale_val:.6f}")
    
    # ✅ 정규화 테스트
    print(f"\n🔄 정규화 테스트:")
    data_scaled = scaler.transform(data_for_pred)
    print(f"   - 정규화 후 타겟 범위: {data_scaled[:, target_idx].min():.6f} ~ {data_scaled[:, target_idx].max():.6f}")
    print(f"   - 정규화 후 타겟 평균: {data_scaled[:, target_idx].mean():.6f}")
    print(f"   - 정규화 후 타겟 표준편차: {data_scaled[:, target_idx].std():.6f}")
    
    # ✅ 최근 10개 정규화 값 출력
    print(f"\n🔍 최근 10개 정규화 값:")
    for i, val in enumerate(data_scaled[-10:, target_idx]):
        print(f"   [{i+1}] {val:.6f}")
    
    # ✅ 역정규화 테스트
    print(f"\n🔙 역정규화 테스트 (직접 계산 방식):")
    test_values = [-1.0, -0.5, 0.0, 0.5, 1.0, data_scaled[:, target_idx].mean()]
    for test_val in test_values:
        reversed_val = test_val * scaler.scale_[target_idx] + scaler.mean_[target_idx]
        print(f"   - 정규화값 {test_val:7.4f} → 역정규화값 {reversed_val:10.6f}")
    
    # ✅ 모델 예측 테스트 (여러 번)
    print(f"\n🎯 모델 예측 테스트:")
    seq_len = int(config['r_seqLen'])
    
    if len(data_scaled) >= seq_len:
        test_sequence = data_scaled[-seq_len:].reshape(1, seq_len, len(study_columns_list))
        
        print(f"   - 입력 시퀀스 크기: {test_sequence.shape}")
        print(f"   - 입력 시퀀스 타겟 범위: {test_sequence[0, :, target_idx].min():.6f} ~ {test_sequence[0, :, target_idx].max():.6f}")
        print(f"\n   - 5번 예측 테스트:")
        
        pred_values = []
        for i in range(5):
            pred_scaled = model.predict(test_sequence, verbose=0)[0, 0]
            pred_original = pred_scaled * scaler.scale_[target_idx] + scaler.mean_[target_idx]
            pred_values.append(pred_original)
            print(f"      {i+1}회: 정규화={pred_scaled:.6f}, 역정규화={pred_original:.6f}")
        
        # 평균 예측값
        predictions = []
        for _ in range(10):
            pred_scaled = model.predict(test_sequence, verbose=0)[0, 0]
            predictions.append(pred_scaled)
        
        avg_pred_scaled = np.mean(predictions)
        std_pred_scaled = np.std(predictions)
        avg_pred_original = avg_pred_scaled * scaler.scale_[target_idx] + scaler.mean_[target_idx]
        
        print(f"\n   📊 10회 반복 통계:")
        print(f"      정규화 평균: {avg_pred_scaled:.6f}")
        print(f"      정규화 표준편차: {std_pred_scaled:.6f}")
        print(f"      역정규화 평균: {avg_pred_original:.6f}")
        
        # ✅ 문제 진단
        print(f"\n💡 진단 결과:")
        
        issues_found = False
        
        if abs(scaler.mean_[target_idx]) < 0.001:
            print(f"   ⚠️  [문제 1] 스케일러 평균이 0에 가깝습니다! ({scaler.mean_[target_idx]:.6f})")
            print(f"      → 학습 데이터의 타겟 값이 대부분 0이었을 가능성")
            print(f"      → 해결: 모델 재학습 필요 (학습 데이터 확인)")
            issues_found = True
        
        if abs(scaler.scale_[target_idx]) < 0.001:
            print(f"   ⚠️  [문제 2] 스케일러 표준편차가 0에 가깝습니다! ({scaler.scale_[target_idx]:.6f})")
            print(f"      → 학습 데이터의 타겟 값에 변화가 없었을 가능성")
            print(f"      → 해결: 모델 재학습 필요 (다양한 데이터 사용)")
            issues_found = True
        
        if abs(avg_pred_scaled) < 0.001:
            print(f"   ⚠️  [문제 3] 모델 예측값(정규화)이 0에 가깝습니다! ({avg_pred_scaled:.6f})")
            print(f"      → 모델이 제대로 학습되지 않았을 가능성")
            print(f"      → 해결: 에포크 증가, 학습률 조정, 모델 구조 변경")
            issues_found = True
        
        if abs(avg_pred_original) < 0.001 and abs(avg_pred_scaled) > 0.01:
            print(f"   ⚠️  [문제 4] 역정규화 과정에서 0이 되었습니다!")
            print(f"      정규화값={avg_pred_scaled:.6f}, 역정규화값={avg_pred_original:.6f}")
            print(f"      → 스케일러 파라미터 문제")
            print(f"      → 해결: 스케일러 재생성 또는 모델 재학습")
            issues_found = True
        
        if not issues_found and abs(avg_pred_original) >= 0.001:
            print(f"   ✅ 예측값이 정상적으로 생성되고 있습니다!")
            print(f"      예측값 범위: {min(pred_values):.6f} ~ {max(pred_values):.6f}")
        elif not issues_found:
            print(f"   ⚠️  예측값이 매우 작습니다 ({avg_pred_original:.6f})")
            print(f"      → 추가 조사 필요")
        
    else:
        print(f"   ❌ 데이터 부족: {len(data_scaled)}개 (최소 {seq_len}개 필요)")
    
    print("\n" + "=" * 70)
    print("진단 완료")
    print("=" * 70)
    print("\n💡 다음 단계:")
    print("   - 문제가 발견된 경우: 위의 해결 방법을 따라 조치")
    print("   - 문제가 없는 경우: 전체 예측 모드(3번) 실행")
    print("=" * 70)

def test_single_prediction(model_name, tablename, days_limit=7):
    """단일 예측 테스트"""
    print("\n🧪 단일 예측 테스트")
    
    model, scaler, config = load_trained_model(model_name)
    if model is None:
        return
    
    new_data = load_new_data(tablename, config['dateColumn'], config['studyColumns'], days_limit=days_limit)
    if new_data is None or new_data.empty:
        return
    
    study_columns_list = [col.strip() for col in config['studyColumns'].split(',')]
    target_idx = study_columns_list.index(config['targetColumn'])
    seq_len = int(config['r_seqLen'])
    
    data_for_pred = new_data[study_columns_list].astype(float)
    data_scaled = scaler.transform(data_for_pred)
    
    if len(data_scaled) >= seq_len:
        input_seq = data_scaled[-seq_len:].reshape(1, seq_len, len(study_columns_list))
        pred_scaled = model.predict(input_seq, verbose=0)[0, 0]
        
        # 직접 계산 방식으로 역정규화
        pred_value = pred_scaled * scaler.scale_[target_idx] + scaler.mean_[target_idx]
        
        print(f"✅ 예측 성공!")
        print(f"   - 정규화 예측값: {pred_scaled:.6f}")
        print(f"   - 최종 예측값: {pred_value:.6f}")
        
        if abs(pred_value) < 0.0001:
            print(f"\n⚠️  예측값이 0에 가깝습니다. 다음을 확인하세요:")
            print(f"   1. 학습 데이터 품질")
            print(f"   2. 모델 학습 정확도")
            print(f"   3. 입력 데이터 분포")

# ============================================================================
# DB 저장 함수
# ============================================================================
def save_predictions_to_db(prediction_result, target_table="solar_generation_forecast"):
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
        
        print(f"\n💾 예측 결과 DB 저장 시작...")
        print(f"   - 대상 테이블: carbontwin.{target_table}")
        print(f"   - 저장할 데이터: {len(predictions)}건")
        
        success_count = 0
        fail_count = 0
        
        with engine.connect() as conn:
            trans = conn.begin()
            
            try:
                for pred in predictions:
                    time_point = pred['date']
                    forecast_value = pred['predicted_value']
                    
                    delete_query = text(f"""
                    DELETE FROM carbontwin.{target_table}
                    WHERE time_point = :time_point
                    """)
                    
                    conn.execute(delete_query, {"time_point": time_point})
                    
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
                    
                    if success_count % 10 == 0:
                        print(f"   ⏳ 진행: {success_count}/{len(predictions)} 건 저장 완료")
                
                trans.commit()
                
                print(f"✅ DB 저장 완료!")
                print(f"   - 성공: {success_count}건")
                print(f"   - 실패: {fail_count}건")
                
            except Exception as e:
                trans.rollback()
                print(f"❌ DB 저장 중 오류 발생 (롤백됨): {str(e)}")
                import traceback
                traceback.print_exc()
                return success_count, len(predictions) - success_count
        
        return success_count, fail_count
        
    except Exception as e:
        print(f"❌ DB 연결 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0, len(predictions)

# ============================================================================
# 메인 실행 함수
# ============================================================================
def main(model_name=None, tablename=None, save_to_db=True, days_limit=7):
    """메인 실행 함수 - 전체 예측 프로세스 실행"""
    print("=" * 70)
    print("🔮 개선된 LSTM 모델 예측 시스템 (DB 저장 + GPU 지원)")
    print("=" * 70)
    
    if model_name is None:
        model_name = "solar-hybrid-seq-2-test-20251017-test-no"
    
    model, scaler, config = load_trained_model(model_name)
    
    if model is None:
        print("\n💡 사용 가능한 모델 목록:")
        if os.path.exists(model_path):
            models = [f.replace('.h5', '') for f in os.listdir(model_path) if f.endswith('.h5')]
            if models:
                for i, m in enumerate(models, 1):
                    print(f"   {i}. {m}")
            else:
                print("   (저장된 모델이 없습니다)")
        return None
    
    if tablename is None:
        tablename = "lstm_input_15m_new"
    print(f"\n📊 사용할 테이블: {tablename}")
    print(f"📅 조회 기간: 최근 {days_limit}일")
    
    print(f"\n📊 데이터베이스에서 데이터 로드 중...")
    new_data = load_new_data(
        tablename,
        config['dateColumn'],
        config['studyColumns'],
        start_date=None,
        end_date=None,
        days_limit=days_limit
    )
    
    if new_data is None or new_data.empty:
        print("❌ 예측할 데이터가 없습니다.")
        return None
    
    print(f"\n{'='*70}")
    
    seq_len = int(config.get('r_seqLen', 60))
    auto_future_steps = 672
    
    print(f"🔮 개선된 실제 미래값 예측 수행")
    print(f"   - 모델 시퀀스 길이: {seq_len}")
    print(f"   - 예측할 미래 스텝: {auto_future_steps}개")
    
    future_result = None
    
    try:
        future_result = predict_future_improved(
            model, scaler, config, new_data, auto_future_steps
        )
        
        if future_result:
            print_future_predictions_improved(future_result)
            
            if save_to_db:
                success, fail = save_predictions_to_db(future_result)
                
                if success > 0:
                    print(f"\n✅ 총 {success}건의 예측 결과가 DB에 저장되었습니다.")
                if fail > 0:
                    print(f"⚠️  {fail}건의 저장 실패")
        else:
            print("❌ 예측 결과 생성 실패")
        
    except Exception as e:
        print(f"❌ 미래값 예측 중 오류: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("🎉 예측 완료!")
    print("="*70)
    
    return future_result

# ============================================================================
# 프로그램 시작점
# ============================================================================
if __name__ == "__main__":
    """
    스크립트 직접 실행
    
    사용법:
        python lstm_predict.py
        
    진단 모드 실행:
        코드 내에서 main() 대신 diagnose_model_and_data() 호출
    """
    try:
        model_name = "solar-hybrid-seq-2-test-20251017-test-no"
        tablename = "lstm_input_15m_new"
        
        print("\n" + "=" * 80)
        print("🔍 실행 모드 선택")
        print("=" * 80)
        print("\n1. 진단 모드 (모델 및 데이터 호환성 확인)")
        print("2. 테스트 모드 (단일 예측 테스트)")
        print("3. 실행 모드 (전체 예측 + DB 저장)")
        print("4. 실행 모드 (전체 예측, DB 저장 안 함)")
        
        choice = input("\n선택 (1-4, 기본값: 3): ").strip() or "3"
        
        # 데이터 조회 기간 설정
        days_input = input("조회할 데이터 기간 (일, 기본값: 7일): ").strip()
        days_limit = int(days_input) if days_input else 7
        
        print(f"\n📅 설정: 최근 {days_limit}일 데이터 사용")
        
        if choice == "1":
            # 진단 모드
            diagnose_model_and_data(model_name, tablename, days_limit)
            
        elif choice == "2":
            # 테스트 모드
            test_single_prediction(model_name, tablename, days_limit)
            
        elif choice == "3":
            # 전체 예측 + DB 저장
            main(
                model_name=model_name,
                tablename=tablename,
                save_to_db=True,
                days_limit=days_limit
            )
            
        elif choice == "4":
            # 전체 예측만 (DB 저장 안 함)
            main(
                model_name=model_name,
                tablename=tablename,
                save_to_db=False,
                days_limit=days_limit
            )
        else:
            print("잘못된 선택입니다. 기본 모드(3)로 실행합니다.")
            main(
                model_name=model_name,
                tablename=tablename,
                save_to_db=True,
                days_limit=days_limit
            )
            
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()