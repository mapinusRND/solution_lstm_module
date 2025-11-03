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
Version : 2.2
Date    : 2025-10-16
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
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta

# ============================================================================
# GPU 설정 및 확인
# ============================================================================
def setup_gpu():
    """
    GPU 설정 및 사용 가능 여부 확인
    
    Returns:
    --------
    bool : GPU 사용 가능 여부
    """
    print("\n" + "=" * 70)
    print("🎮 GPU 설정 확인")
    print("=" * 70)
    
    # TensorFlow 버전 출력
    print(f"📌 TensorFlow 버전: {tf.__version__}")
    
    # GPU 디바이스 목록 확인
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # ================================================================
            # GPU 메모리 동적 할당 설정 (권장)
            # ================================================================
            # GPU 메모리를 한 번에 모두 할당하지 않고, 필요한 만큼만 동적으로 할당
            # 여러 프로세스가 GPU를 공유할 때 유용
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # GPU 개수 및 이름 출력
            print(f"✅ GPU 사용 가능: {len(gpus)}개")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
                # GPU 메모리 정보 (가능한 경우)
                try:
                    gpu_details = tf.config.experimental.get_device_details(gpu)
                    if 'device_name' in gpu_details:
                        print(f"        모델명: {gpu_details['device_name']}")
                except:
                    pass
            
            # CUDA 및 cuDNN 버전 확인
            build_info = tf.sysconfig.get_build_info()
            print(f"   CUDA 버전: {build_info.get('cuda_version', 'N/A')}")
            print(f"   cuDNN 버전: {build_info.get('cudnn_version', 'N/A')}")
            
            # 논리적 GPU 목록 (메모리 제한 설정 후)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"   논리적 GPU: {len(logical_gpus)}개")
            
            print("\n💡 GPU 가속이 활성화되었습니다!")
            return True
            
        except RuntimeError as e:
            # GPU 설정 중 오류 발생
            print(f"❌ GPU 설정 오류: {e}")
            print("⚠️  CPU 모드로 실행됩니다.")
            return False
    else:
        # GPU를 찾을 수 없는 경우
        print("⚠️  사용 가능한 GPU를 찾을 수 없습니다.")
        print("💡 CPU 모드로 실행됩니다.")
        print("\n📝 GPU를 사용하려면:")
        print("   1. NVIDIA GPU 드라이버 설치")
        print("   2. CUDA Toolkit 설치 (11.8 또는 12.x)")
        print("   3. cuDNN 설치")
        print("   4. TensorFlow GPU 버전 설치: pip install tensorflow[and-cuda]")
        return False

# GPU 설정 실행
gpu_available = setup_gpu()

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
        조회할 테이블명 (예: 'lstm_input_15m_new')
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
    GPU가 사용 가능한 경우 자동으로 GPU에서 실행됨
    
    Parameters:
    -----------
    model_name : str
        로드할 모델명 (예: 'solar-hybrid-seq-2-test-20251017-test-no')
        
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
        
        # GPU가 사용 가능한 경우 GPU에서 모델 로드
        if gpu_available:
            with tf.device('/GPU:0'):  # 첫 번째 GPU 사용
                model = load_model(model_file, compile=False)
                model.compile(optimizer='adam', loss='mse')
                print(f"   🎮 GPU에 모델 로드 완료")
        else:
            # CPU에서 모델 로드
            model = load_model(model_file, compile=False)
            model.compile(optimizer='adam', loss='mse')
            print(f"   💻 CPU에 모델 로드 완료")
        
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
        print(f"   - 날짜 컬럼 : {config['dateColumn']}")  # 입력 변수들
        print(f"   - 시퀀스 길이: {config['r_seqLen']}")  # LSTM 입력 시퀀스 길이
        print(f"   - 예측 일수: {config['r_predDays']}")  # 몇 스텝 앞을 예측하는지
        
        return model, scaler, config
        
    except Exception as e:
        print(f"❌ 모델 로드 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

# ============================================================================
# 🔥 개선된 미래값 예측 (중복 예측 문제 해결 + GPU 가속)
# ============================================================================
def predict_future_improved(model, scaler, config, new_data, future_steps=None):
    """
    개선된 미래값 예측 - 재귀적 예측으로 실제 미래값 생성
    GPU가 사용 가능한 경우 자동으로 GPU에서 예측 수행
    
    개선사항:
    1. 시간 정보 추가 (시간, 분) - 태양광 발전은 시간대별 패턴이 중요
    2. 더 다양한 노이즈 추가 - 예측의 다양성 확보
    3. 예측값 범위 검증 - 물리적 제약 조건 적용 (야간=0)
    4. 앙상블 예측 - 여러 번 예측하여 평균 (안정성 향상)
    5. GPU 가속 지원 - 예측 속도 향상
    
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
        if gpu_available:
            print(f"   - 실행 환경: 🎮 GPU 가속")
        else:
            print(f"   - 실행 환경: 💻 CPU")
        
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
        
        # ====================================================================
        # 성능 측정을 위한 시간 기록
        # ====================================================================
        import time
        start_time = time.time()
        prediction_times = []  # 각 스텝별 예측 시간 기록
        
        # 재귀적 예측 루프 (각 미래 스텝마다 반복)
        for step in range(future_steps):
            step_start_time = time.time()  # 현재 스텝 시작 시간
            # 다음 예측 시점 계산
            next_date = last_date + time_delta * (step + 1)
            
            # 시간 정보 추출 (태양광 발전은 시간대가 중요)
            hour = next_date.hour
            minute = next_date.minute
            
            # 🔥 앙상블 예측: 여러 번 예측하여 평균 (안정성 향상)
            ensemble_predictions = []
            
            for _ in range(n_ensemble):
                # 노이즈 추가 (입력 데이터에 작은 변동 추가)
                noisy_sequence = current_sequence + np.random.normal(0, 0.05, current_sequence.shape)
                
                # LSTM 입력 형태로 변환: (batch_size=1, seq_len, features)
                input_data = noisy_sequence.reshape(1, seq_len, len(study_columns_list))
                
                # 🎮 GPU에서 모델 예측 (GPU 사용 가능 시)
                # verbose=0: 예측 진행 상황 출력 안 함
                pred_scaled = model.predict(input_data, verbose=0)
                ensemble_predictions.append(pred_scaled[0, 0])
            
            # 앙상블 평균 및 표준편차 계산
            avg_pred_scaled = np.mean(ensemble_predictions)
            pred_std = np.std(ensemble_predictions)
            
            # 신뢰도 계산
            distance_penalty = 1.0 - (step / future_steps) * 0.3
            ensemble_uncertainty = min(pred_std / 0.1, 1.0)
            confidence = distance_penalty * (1.0 - ensemble_uncertainty)
            confidence = max(0.0, min(1.0, confidence))
            
            # 예측값 역정규화
            mean_values = scaler.mean_.copy()
            mean_values[target_idx] = avg_pred_scaled
            pred_value = scaler.inverse_transform([mean_values])[0, target_idx]
            
            # 🔥 태양광 발전량 물리적 제약 적용
            if 18 <= hour or hour < 6:
                pred_value = max(0, pred_value * 0.1)  # 야간
            else:
                pred_value = max(0, pred_value)  # 주간
            
            # 결과 저장
            future_predictions.append(pred_value)
            future_dates.append(next_date)
            prediction_confidence.append(confidence)
            
            # 🔥 다음 시퀀스 준비
            new_point = current_sequence[-1].copy()
            new_point[target_idx] = avg_pred_scaled
            
            time_factor = np.sin(2 * np.pi * hour / 24)
            for i in range(len(new_point)):
                if i != target_idx:
                    new_point[i] += np.random.normal(0, 0.02) * time_factor
            
            current_sequence = np.vstack([current_sequence[1:], new_point])
            
            # 현재 스텝 완료 시간 기록
            step_elapsed = time.time() - step_start_time
            prediction_times.append(step_elapsed)
            
            # 진행상황 표시 (성능 정보 포함)
            if (step + 1) % 10 == 0 or step == future_steps - 1:
                avg_time_per_step = sum(prediction_times) / len(prediction_times)
                print(f"   ⏳ 진행: {step + 1}/{future_steps} 스텝 완료 "
                      f"(평균 {avg_time_per_step*1000:.1f}ms/스텝)")
        
        # ====================================================================
        # 예측 완료 시간 계산 및 성능 통계
        # ====================================================================
        elapsed_time = time.time() - start_time
        avg_step_time = sum(prediction_times) / len(prediction_times) if prediction_times else 0
        min_step_time = min(prediction_times) if prediction_times else 0
        max_step_time = max(prediction_times) if prediction_times else 0
        
        # 성능 정보 출력
        print(f"\n✅ 미래값 예측 완료!")
        print(f"📊 성능 통계:")
        print(f"   - 총 소요 시간: {elapsed_time:.3f}초")
        print(f"   - 평균 스텝 시간: {avg_step_time*1000:.2f}ms")
        print(f"   - 최소 스텝 시간: {min_step_time*1000:.2f}ms")
        print(f"   - 최대 스텝 시간: {max_step_time*1000:.2f}ms")
        print(f"   - 처리량: {future_steps/elapsed_time:.2f} 스텝/초")
        
        if gpu_available:
            print(f"   🎮 GPU 가속 활성화")
        else:
            print(f"   💻 CPU 모드")
            print(f"   💡 GPU 사용 시 약 5-20배 빠를 수 있습니다")
        
        # 결과 구성
        future_result = {
            "model_name": config['modelName'],
            "target_column": targetColumn,
            "prediction_type": "future_improved",
            "base_date": last_date.isoformat(),
            "sequence_length": seq_len,
            "future_steps": future_steps,
            "prediction_interval": pred_days,
            "gpu_used": gpu_available,
            "performance": {
                "total_time_seconds": round(elapsed_time, 3),
                "average_step_time_ms": round(avg_step_time * 1000, 2),
                "min_step_time_ms": round(min_step_time * 1000, 2),
                "max_step_time_ms": round(max_step_time * 1000, 2),
                "throughput_steps_per_sec": round(future_steps / elapsed_time, 2)
            },
            "predictions": []
        }
        
        # 각 스텝별 예측 결과 저장
        for i, (date, pred, conf) in enumerate(zip(future_dates, future_predictions, prediction_confidence)):
            future_result["predictions"].append({
                "step": i + 1,
                "date": date.isoformat(),
                "predicted_value": convert_to_serializable(pred),
                "confidence": convert_to_serializable(conf),
                "hour": date.hour,
                "is_daytime": 6 <= date.hour < 18
            })
        
        # 통계 정보 추가
        future_result["statistics"] = {
            "min_predicted": convert_to_serializable(np.min(future_predictions)),
            "max_predicted": convert_to_serializable(np.max(future_predictions)),
            "mean_predicted": convert_to_serializable(np.mean(future_predictions)),
            "std_predicted": convert_to_serializable(np.std(future_predictions)),
            "avg_confidence": convert_to_serializable(np.mean(prediction_confidence))
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
    performance = result.get('performance', {})
    
    # 헤더 출력
    print(f"\n🔮 개선된 미래값 예측 결과:")
    print(f"   기준 시점: {result['base_date'][:19]}")
    print(f"   시퀀스 길이: {result.get('sequence_length', 'N/A')}개")
    print(f"   총 예측 스텝: {result['future_steps']}개")
    
    # 성능 정보 출력
    print(f"\n⚡ 성능 정보:")
    print(f"   실행 환경: {'🎮 GPU' if result.get('gpu_used', False) else '💻 CPU'}")
    print(f"   총 소요 시간: {performance.get('total_time_seconds', 0):.3f}초")
    print(f"   평균 스텝 시간: {performance.get('average_step_time_ms', 0):.2f}ms")
    print(f"   처리 속도: {performance.get('throughput_steps_per_sec', 0):.2f} 스텝/초")
    
    # GPU 사용 시 예상 성능 향상 정보
    if not result.get('gpu_used', False):
        estimated_gpu_time = performance.get('total_time_seconds', 0) / 10  # 약 10배 빠를 것으로 예상
        print(f"\n   💡 GPU 사용 시 예상 시간: ~{estimated_gpu_time:.3f}초 (약 5-20배 향상)")
    
    print("\n" + "=" * 80)
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

# ============================================================================
# 🔥 미래 예측 결과를 DB에 저장
# ============================================================================
def save_predictions_to_db(prediction_result, target_table="solar_generation_forecast"):
    """
    미래 예측 결과를 PostgreSQL DB에 저장
    time_point가 중복되면 기존 데이터 삭제 후 신규 데이터 INSERT
    
    Parameters:
    -----------
    prediction_result : dict
        predict_future_improved() 함수의 반환값
    target_table : str
        저장할 테이블명 (기본값: 'solar_generation_forecast')
        
    Returns:
    --------
    tuple : (성공 건수, 실패 건수)
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
                    
                    # 기존 데이터 삭제
                    delete_query = text(f"""
                    DELETE FROM carbontwin.{target_table}
                    WHERE time_point = :time_point
                    """)
                    
                    conn.execute(delete_query, {"time_point": time_point})
                    
                    # 새로운 데이터 INSERT
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
def main(model_name=None, tablename=None, save_to_db=True):
    """
    메인 실행 함수 - 전체 예측 프로세스 실행
    
    Parameters:
    -----------
    model_name : str, optional
        사용할 모델명 (None이면 기본값 사용)
    tablename : str, optional
        데이터를 가져올 테이블명 (None이면 기본값 사용)
    save_to_db : bool, optional
        예측 결과를 DB에 저장할지 여부 (기본값: True)
        
    Returns:
    --------
    dict : 미래 예측 결과
    """
    print("=" * 70)
    print("🔮 개선된 LSTM 모델 예측 시스템 (DB 저장 + GPU 지원)")
    print("=" * 70)
    
    # 1. 모델명
    if model_name is None:
        model_name = "solar-hybrid-seq-2-test-20251017-test-no"
    
    # 2. 모델, 스케일러, 설정 로드
    model, scaler, config = load_trained_model(model_name)
    
    # 모델 로드 실패 시
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
    
    # 3. 테이블명 설정
    if tablename is None:
        tablename = "lstm_input_15m_new"
    print(f"\n📊 사용할 테이블: {tablename}")
    
    # 4. DB에서 데이터 로드
    print(f"\n📊 데이터베이스에서 데이터 로드 중...")
    new_data = load_new_data(
        tablename,
        config['dateColumn'],
        config['studyColumns'],
        start_date=None,
        end_date=None
    )
    
    if new_data is None or new_data.empty:
        print("❌ 예측할 데이터가 없습니다.")
        return None
    
    # 5. 실제 미래값 예측 실행
    print(f"\n{'='*70}")
    
    seq_len = int(config.get('r_seqLen', 60))
    auto_future_steps = 672
    
    print(f"🔮 개선된 실제 미래값 예측 수행")
    print(f"   - 모델 시퀀스 길이: {seq_len}")
    print(f"   - 예측할 미래 스텝: {auto_future_steps}개")
    
    future_result = None
    
    try:
        # 미래값 예측 수행 (GPU 가속 지원)
        future_result = predict_future_improved(
            model, scaler, config, new_data, auto_future_steps
        )
        
        if future_result:
            # 예측 결과 콘솔 출력
            print_future_predictions_improved(future_result)
            
            # DB에 저장
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
    
    # 6. 완료 메시지
    print(f"\n{'='*70}")
    print("🎉 예측 완료!")
    print("="*70)
    
    return future_result

# ============================================================================
# 프로그램 시작점
# ============================================================================
if __name__ == "__main__":
    """
    스크립트 직접 실행 시 main() 함수 호출
    
    사용법:
        # DB 저장 활성화 (기본)
        python lstm_predict.py
        
        # DB 저장 비활성화 (테스트용)
        # main(save_to_db=False) 형태로 코드 수정 필요
        
    GPU 사용 요구사항:
        1. NVIDIA GPU 드라이버 설치
        2. CUDA Toolkit 11.8 또는 12.x 설치
        3. cuDNN 8.x 설치
        4. TensorFlow GPU: pip install tensorflow[and-cuda]
    """
    try:
        # DB 저장 활성화 상태로 실행
        main(
            model_name="solar-hybrid-seq-2-test-20251017-test-no",      # 모델명
            tablename="lstm_input_15m_new", # 입력 테이블명
            save_to_db=True            # DB 저장 활성화
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()