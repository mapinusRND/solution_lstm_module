# -*- coding: utf-8 -*-
"""
Title   : LSTM 모델 예측 전용 모듈 (DB 연동 버전)
Author  : 주성중 / (주)맵인어스
Purpose : 학습된 LSTM 모델로 새로운 데이터 예측 (DB에서 모델 선택)
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import json
import joblib
from tensorflow.keras.models import load_model
from sqlalchemy import create_engine
from datetime import datetime, timedelta

# 환경 설정
ENV = os.getenv('FLASK_ENV', 'local')
if ENV == 'local':
    root = "D:/work/lstm"
else:
    root = "/app/webfiles/lstm"

model_path = os.path.abspath(root + "/saved_models")
prediction_path = os.path.abspath(root + "/predictions")
os.makedirs(prediction_path, exist_ok=True)


def get_db_engine():
    """SQLAlchemy 엔진 생성"""
    connection_string = "postgresql://postgres:mapinus@10.10.10.201:5432/postgres"
    return create_engine(connection_string)


# ============================================================================
# DB 조회 함수들
# ============================================================================

def get_available_models_from_db(show_stats=True):
    """
    DB에서 사용 가능한 모델 목록 조회
    
    Args:
        show_stats: 모델별 최고 성능 통계 포함 여부
    
    Returns:
        DataFrame: 모델 목록
    """
    try:
        engine = get_db_engine()
        
        if show_stats:
            query = """
            SELECT 
                m.model_id,
                m.model_name,
                m.target_column,
                m.epochs,
                m.sequence_length,
                m.prediction_days,
                m.created_at as model_created_at,
                COUNT(e.experiment_id) as total_experiments,
                MAX(e.accuracy) as best_accuracy,
                MIN(e.mape) as best_mape,
                MAX(e.r2_score) as best_r2_score,
                MAX(e.created_at) as last_experiment_date
            FROM carbontwin.lstm_model m
            LEFT JOIN carbontwin.lstm_experiment e ON m.model_id = e.model_id
            WHERE e.status = 'success' OR e.status IS NULL
            GROUP BY m.model_id, m.model_name, m.target_column, m.epochs, 
                     m.sequence_length, m.prediction_days, m.created_at
            ORDER BY best_accuracy DESC NULLS LAST, m.created_at DESC
            """
        else:
            query = """
            SELECT 
                model_id,
                model_name,
                target_column,
                date_column,
                study_columns,
                epochs,
                batch_size,
                validation_split,
                sequence_length,
                prediction_days,
                created_at
            FROM carbontwin.lstm_model
            ORDER BY created_at DESC
            """
        
        models = pd.read_sql_query(query, engine)
        return models
        
    except Exception as e:
        print(f"❌ 모델 목록 조회 실패: {str(e)}")
        return None


def get_model_by_id(model_id):
    """model_id로 모델 정보 조회"""
    try:
        engine = get_db_engine()
        
        query = f"""
        SELECT *
        FROM carbontwin.lstm_model
        WHERE model_id = {model_id}
        """
        
        result = pd.read_sql_query(query, engine)
        
        if result.empty:
            print(f"❌ model_id {model_id}를 찾을 수 없습니다.")
            return None
        
        return result.iloc[0].to_dict()
        
    except Exception as e:
        print(f"❌ 모델 조회 실패: {str(e)}")
        return None


def get_model_by_name(model_name):
    """model_name으로 모델 정보 조회"""
    try:
        engine = get_db_engine()
        
        query = f"""
        SELECT *
        FROM carbontwin.lstm_model
        WHERE model_name = '{model_name}'
        """
        
        result = pd.read_sql_query(query, engine)
        
        if result.empty:
            print(f"❌ 모델 '{model_name}'을 찾을 수 없습니다.")
            return None
        
        return result.iloc[0].to_dict()
        
    except Exception as e:
        print(f"❌ 모델 조회 실패: {str(e)}")
        return None


def get_best_experiment_for_model(model_id, metric='accuracy'):
    """
    특정 모델의 최고 성능 실험 조회
    
    Args:
        model_id: 모델 ID
        metric: 정렬 기준 ('accuracy', 'mape', 'r2_score', 'rmse')
    
    Returns:
        dict: 최고 성능 실험 정보
    """
    try:
        engine = get_db_engine()
        
        order = 'DESC' if metric in ['accuracy', 'r2_score'] else 'ASC'
        
        query = f"""
        SELECT 
            e.*,
            m.model_name,
            m.target_column,
            m.study_columns,
            m.date_column
        FROM carbontwin.lstm_experiment e
        JOIN carbontwin.lstm_model m ON e.model_id = m.model_id
        WHERE e.model_id = {model_id} AND e.status = 'success'
        ORDER BY e.{metric} {order}
        LIMIT 1
        """
        
        result = pd.read_sql_query(query, engine)
        
        if result.empty:
            print(f"⚠️ model_id {model_id}에 대한 성공한 실험이 없습니다.")
            return None
        
        return result.iloc[0].to_dict()
        
    except Exception as e:
        print(f"❌ 실험 조회 실패: {str(e)}")
        return None


def get_experiments_by_model(model_id, limit=10):
    """특정 모델의 실험 이력 조회"""
    try:
        engine = get_db_engine()
        
        query = f"""
        SELECT 
            experiment_id,
            experiment_name,
            accuracy,
            mape,
            rmse,
            r2_score,
            execution_time_seconds,
            status,
            created_at
        FROM carbontwin.lstm_experiment
        WHERE model_id = {model_id}
        ORDER BY created_at DESC
        LIMIT {limit}
        """
        
        return pd.read_sql_query(query, engine)
        
    except Exception as e:
        print(f"❌ 실험 이력 조회 실패: {str(e)}")
        return None


# ============================================================================
# 모델 로드 함수 (DB 연동)
# ============================================================================

def load_trained_model_from_db(model_id=None, model_name=None, use_best_experiment=True):
    """
    DB에서 모델 정보를 조회하여 로드
    
    Args:
        model_id: 모델 ID (우선순위 높음)
        model_name: 모델 이름
        use_best_experiment: 최고 성능 실험의 파일 경로 사용 여부
    
    Returns:
        dict: {
            'model': Keras 모델,
            'scaler': StandardScaler,
            'config': 설정 정보,
            'model_info': DB 모델 정보,
            'experiment_info': 실험 정보 (있을 경우)
        }
    """
    try:
        # 모델 정보 조회
        if model_id:
            model_info = get_model_by_id(model_id)
        elif model_name:
            model_info = get_model_by_name(model_name)
        else:
            raise ValueError("❌ model_id 또는 model_name을 지정해야 합니다.")
        
        if not model_info:
            return None
        
        print(f"\n📂 모델 정보:")
        print(f"   - ID: {model_info['model_id']}")
        print(f"   - 이름: {model_info['model_name']}")
        print(f"   - 타겟: {model_info['target_column']}")
        print(f"   - 시퀀스 길이: {model_info['sequence_length']}")
        print(f"   - Epochs: {model_info['epochs']}")
        
        # 최고 성능 실험 조회
        experiment_info = None
        if use_best_experiment:
            experiment_info = get_best_experiment_for_model(
                model_info['model_id'], 
                metric='accuracy'
            )
            
            if experiment_info:
                print(f"\n🏆 최고 성능 실험:")
                print(f"   - 실험명: {experiment_info['experiment_name']}")
                print(f"   - 정확도: {experiment_info['accuracy']:.2f}%")
                print(f"   - MAPE: {experiment_info['mape']:.2f}%")
                print(f"   - R² Score: {experiment_info['r2_score']:.4f}")
        
        # 파일 경로 결정
        model_name_for_file = model_info['model_name']
        
        # 실험에서 model_file_path가 있으면 사용
        if experiment_info and experiment_info.get('model_file_path'):
            model_file = experiment_info['model_file_path']
        else:
            model_file = os.path.join(model_path, f"{model_name_for_file}.h5")
        
        scaler_file = os.path.join(model_path, f"{model_name_for_file}_scaler.pkl")
        config_file = os.path.join(model_path, f"{model_name_for_file}_config.json")
        
        # 파일 존재 확인
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"❌ 모델 파일을 찾을 수 없습니다: {model_file}")
        if not os.path.exists(scaler_file):
            raise FileNotFoundError(f"❌ 스케일러 파일을 찾을 수 없습니다: {scaler_file}")
        
        # 모델 로드
        print(f"\n📥 파일 로딩 중...")
        model = load_model(model_file, compile=False)
        scaler = joblib.load(scaler_file)
        print(f"✅ 모델 및 스케일러 로드 완료")
        
        # 설정 파일 로드 (없으면 DB에서 생성)
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"✅ 설정 파일 로드 완료")
        else:
            # DB 정보로 config 생성
            print(f"⚠️ 설정 파일이 없어 DB 정보로 생성합니다.")
            config = {
                'modelName': model_info['model_name'],
                'dateColumn': model_info['date_column'],
                'studyColumns': model_info['study_columns'],
                'targetColumn': model_info['target_column'],
                'r_epochs': int(model_info['epochs']),
                'r_batchSize': int(model_info['batch_size']),
                'r_validationSplit': float(model_info['validation_split']),
                'r_seqLen': int(model_info['sequence_length']),
                'r_predDays': int(model_info['prediction_days']),
                'tablename': 'lstm_input_1m'  # 기본값
            }
            
            # config_json이 있으면 사용
            if experiment_info and experiment_info.get('config_json'):
                try:
                    config.update(json.loads(experiment_info['config_json']))
                except:
                    pass
        
        print(f"\n📋 설정 정보:")
        print(f"   - 입력 컬럼: {config['studyColumns']}")
        print(f"   - 타겟 컬럼: {config['targetColumn']}")
        print(f"   - 시퀀스 길이: {config['r_seqLen']}")
        print(f"   - 예측 일수: {config['r_predDays']}")
        
        return {
            'model': model,
            'scaler': scaler,
            'config': config,
            'model_info': model_info,
            'experiment_info': experiment_info
        }
        
    except Exception as e:
        print(f"❌ 모델 로드 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# 데이터 로드 함수
# ============================================================================

def load_new_data_from_db(tablename, dateColumn, studyColumns, start_date=None, end_date=None, limit=None, daytime_only=False):
    """
    데이터베이스에서 새로운 데이터를 로드
    
    Args:
        tablename: 테이블명
        dateColumn: 날짜 컬럼명
        studyColumns: 사용할 컬럼들 (쉼표로 구분된 문자열)
        start_date: 시작 날짜 (선택, 'YYYY-MM-DD HH:MM:SS' 형식)
        end_date: 종료 날짜 (선택)
        limit: 최대 로드할 행 수 (선택)
        daytime_only: True면 has_sunlight=1인 주간 데이터만 로드
    """
    try:
        engine = get_db_engine()
        
        # 기본 쿼리
        query = f"""
        SELECT {studyColumns}, {dateColumn}
        FROM carbontwin.{tablename}
        WHERE {dateColumn} IS NOT NULL
        """
        
        # 주간 데이터만 필터링 (has_sunlight 컬럼이 있는 경우)
        if daytime_only and 'has_sunlight' in studyColumns:
            query += " AND has_sunlight = 1"
        
        # 날짜 필터 추가
        if start_date:
            query += f" AND {dateColumn} >= '{start_date}'"
        if end_date:
            query += f" AND {dateColumn} <= '{end_date}'"
        
        query += f" ORDER BY {dateColumn} ASC"
        
        # 제한 추가
        if limit:
            query += f" LIMIT {limit}"
        
        data = pd.read_sql_query(query, engine)
        
        if daytime_only and 'has_sunlight' in studyColumns:
            print(f"✅ 데이터 로드 완료: {len(data)}행 (주간 데이터만)")
        else:
            print(f"✅ 데이터 로드 완료: {len(data)}행")
        
        if len(data) == 0:
            print("⚠️ 데이터가 비어있습니다.")
            return None
            
        return data
        
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {str(e)}")
        return None


# ============================================================================
# 예측 함수
# ============================================================================

def predict_with_model(model_info, new_data):
    """
    로드된 모델로 새로운 데이터 예측
    
    Args:
        model_info: load_trained_model_from_db()의 반환값
        new_data: 예측할 데이터 (DataFrame)
    
    Returns:
        dict: 예측 결과
    """
    try:
        model = model_info['model']
        scaler = model_info['scaler']
        config = model_info['config']
        
        # 설정 추출
        dateColumn = config['dateColumn']
        studyColumns = config['studyColumns']
        targetColumn = config['targetColumn']
        seq_len = int(config['r_seqLen'])
        pred_days = int(config['r_predDays'])
        
        study_columns_list = [col.strip() for col in studyColumns.split(',')]
        target_idx = study_columns_list.index(targetColumn)
        
        print(f"\n🔮 예측 시작...")
        print(f"   - 시퀀스 길이: {seq_len}")
        print(f"   - 예측 일수: {pred_days}")
        print(f"   - 입력 데이터 수: {len(new_data)}")
        
        # 날짜 처리
        if dateColumn in new_data.columns:
            dates = pd.to_datetime(new_data[dateColumn], errors='coerce')
        else:
            dates = pd.date_range(start='2025-01-01', periods=len(new_data), freq='1min')
            print(f"⚠️ 날짜 컬럼이 없어 가상 날짜를 생성했습니다.")
        
        # 데이터 전처리
        data_for_prediction = new_data[study_columns_list].astype(float)
        
        # 스케일링 (학습 시 사용한 스케일러 사용!)
        data_scaled = scaler.transform(data_for_prediction)
        
        # 시퀀스 생성 확인
        if len(data_scaled) < seq_len + pred_days - 1:
            raise ValueError(
                f"❌ 데이터가 부족합니다. 최소 {seq_len + pred_days - 1}개 필요, 현재 {len(data_scaled)}개"
            )
        
        # 예측용 시퀀스 생성
        predictX = []
        valid_indices = []
        
        for i in range(seq_len, len(data_scaled) - pred_days + 1):
            predictX.append(data_scaled[i - seq_len:i, :])
            valid_indices.append(i + pred_days - 1)
        
        predictX = np.array(predictX)
        print(f"   - 예측 가능한 시퀀스: {len(predictX)}개")
        
        # 예측 수행
        print(f"\n⏳ 예측 진행 중...")
        predictions_scaled = model.predict(predictX, verbose=0)
        print(f"✅ 예측 완료!")
        
        # 역변환 (원래 스케일로 복원)
        mean_values = np.repeat(scaler.mean_[np.newaxis, :], predictions_scaled.shape[0], axis=0)
        mean_values[:, target_idx] = np.squeeze(predictions_scaled)
        predictions_original = scaler.inverse_transform(mean_values)[:, target_idx]
        
        # ✅ 태양광 발전량은 음수가 나올 수 없으므로 0 이상으로 클리핑
        predictions_original = np.maximum(predictions_original, 0)
        print(f"   - 음수 예측값을 0으로 보정 완료")
        
        # 예측 날짜
        prediction_dates = dates.iloc[valid_indices].reset_index(drop=True)
        
        # 실제값이 있다면 비교
        actual_values = data_for_prediction[targetColumn].iloc[valid_indices].values
        has_actual = not np.all(pd.isna(actual_values))
        
        # 결과 구성
        results = []
        for i in range(len(predictions_original)):
            result_record = {
                "index": i,
                "date": prediction_dates.iloc[i].isoformat() if hasattr(prediction_dates.iloc[i], 'isoformat') else str(prediction_dates.iloc[i]),
                "predicted_value": float(predictions_original[i])
            }
            
            if has_actual and not pd.isna(actual_values[i]):
                result_record["actual_value"] = float(actual_values[i])
                result_record["difference"] = float(predictions_original[i] - actual_values[i])
                result_record["percentage_error"] = float(
                    abs((predictions_original[i] - actual_values[i]) / actual_values[i] * 100) 
                    if actual_values[i] != 0 else 0
                )
            else:
                result_record["actual_value"] = None
                result_record["difference"] = None
                result_record["percentage_error"] = None
            
            results.append(result_record)
        
        # 통계 계산
        statistics = {
            "predicted_min": float(np.min(predictions_original)),
            "predicted_max": float(np.max(predictions_original)),
            "predicted_mean": float(np.mean(predictions_original)),
            "predicted_std": float(np.std(predictions_original))
        }
        
        if has_actual:
            valid_actuals = actual_values[~pd.isna(actual_values)]
            valid_preds = predictions_original[:len(valid_actuals)]
            
            # ✅ MAPE 계산 시 실제값이 0이 아닌 것만 사용 (division by zero 방지)
            non_zero_mask = valid_actuals != 0
            
            if np.sum(non_zero_mask) > 0:
                mape_value = float(np.mean(np.abs((valid_actuals[non_zero_mask] - valid_preds[non_zero_mask]) / valid_actuals[non_zero_mask])) * 100)
            else:
                mape_value = None
            
            statistics.update({
                "actual_min": float(np.min(valid_actuals)),
                "actual_max": float(np.max(valid_actuals)),
                "actual_mean": float(np.mean(valid_actuals)),
                "mae": float(np.mean(np.abs(valid_preds - valid_actuals))),
                "rmse": float(np.sqrt(np.mean((valid_preds - valid_actuals) ** 2))),
                "mape": mape_value,
                "non_zero_count": int(np.sum(non_zero_mask)),
                "total_count": len(valid_actuals),
                "zero_count": int(len(valid_actuals) - np.sum(non_zero_mask))
            })
        
        print(f"\n📊 예측 결과 요약:")
        print(f"   - 예측값 범위: {statistics['predicted_min']:.3f} ~ {statistics['predicted_max']:.3f}")
        print(f"   - 예측값 평균: {statistics['predicted_mean']:.3f}")
        
        if has_actual:
            print(f"   - MAE: {statistics['mae']:.4f}")
            print(f"   - RMSE: {statistics['rmse']:.4f}")
            
            if statistics.get('mape') is not None:
                print(f"   - MAPE: {statistics['mape']:.2f}% (실제값 0 제외)")
                print(f"   - 정확도: {100 - statistics['mape']:.2f}%")
            else:
                print(f"   - MAPE: 계산 불가 (모든 실제값이 0)")
            
            print(f"   - 평가 데이터: {statistics['non_zero_count']}개 (0이 아닌 값) / {statistics['total_count']}개 (전체)")
            
            if statistics['zero_count'] > 0:
                print(f"   ⚠️  {statistics['zero_count']}개의 0값 데이터 제외하고 평가됨")
        
        # 모델 정보 추가
        result_data = {
            "status": "success",
            "model_id": model_info['model_info']['model_id'],
            "model_name": config['modelName'],
            "target_column": targetColumn,
            "prediction_count": len(results),
            "timestamp": datetime.now().isoformat(),
            "statistics": statistics,
            "predictions": results
        }
        
        # 실험 정보가 있으면 추가
        if model_info.get('experiment_info'):
            exp = model_info['experiment_info']
            result_data['experiment_info'] = {
                "experiment_id": int(exp['experiment_id']),
                "experiment_name": exp['experiment_name'],
                "training_accuracy": float(exp['accuracy']),
                "training_mape": float(exp['mape']),
                "r2_score": float(exp['r2_score'])
            }
        
        return result_data
        
    except Exception as e:
        print(f"❌ 예측 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e)
        }


def predict_future(model_info, last_data, future_steps=10):
    """
    마지막 데이터를 기반으로 미래 예측 (재귀적 예측)
    
    Args:
        model_info: 로드된 모델 정보
        last_data: 마지막 시퀀스 데이터 (DataFrame, 최소 seqLen 길이)
        future_steps: 예측할 미래 시점 수
    
    Returns:
        list: 미래 예측값들
    """
    try:
        model = model_info['model']
        scaler = model_info['scaler']
        config = model_info['config']
        
        seq_len = int(config['r_seqLen'])
        studyColumns = config['studyColumns']
        targetColumn = config['targetColumn']
        
        study_columns_list = [col.strip() for col in studyColumns.split(',')]
        target_idx = study_columns_list.index(targetColumn)
        
        print(f"\n🔮 미래 예측 시작...")
        print(f"   - 예측 시점 수: {future_steps}")
        
        # 초기 시퀀스 준비
        if len(last_data) < seq_len:
            raise ValueError(f"❌ 최소 {seq_len}개의 데이터가 필요합니다.")
        
        # 마지막 seqLen 개만 사용
        initial_sequence = last_data[study_columns_list].tail(seq_len).astype(float)
        sequence_scaled = scaler.transform(initial_sequence.values)
        
        future_predictions = []
        current_sequence = sequence_scaled.copy()
        
        for step in range(future_steps):
            # 현재 시퀀스로 예측
            input_seq = current_sequence[-seq_len:].reshape(1, seq_len, -1)
            pred_scaled = model.predict(input_seq, verbose=0)
            
            # 역변환
            mean_values = scaler.mean_.copy()
            mean_values[target_idx] = pred_scaled[0, 0]
            pred_original = scaler.inverse_transform(mean_values.reshape(1, -1))[0, target_idx]
            
            future_predictions.append(float(pred_original))
            
            # 다음 시퀀스 준비
            next_point = current_sequence[-1].copy()
            next_point[target_idx] = pred_scaled[0, 0]
            
            # 시퀀스 업데이트
            current_sequence = np.vstack([current_sequence, next_point])
            
            print(f"   Step {step+1}/{future_steps}: {pred_original:.3f}")
        
        print(f"✅ 미래 예측 완료!")
        return future_predictions
        
    except Exception as e:
        print(f"❌ 미래 예측 실패: {str(e)}")
        return None


def save_prediction_results(prediction_result, output_filename=None):
    """예측 결과를 JSON 파일로 저장"""
    try:
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{prediction_result['model_name']}_prediction_{timestamp}.json"
        
        output_path = os.path.join(prediction_path, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(prediction_result, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 예측 결과 저장 완료: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ 결과 저장 실패: {str(e)}")
        return None


# ============================================================================
# 메인 실행부
# ============================================================================
if __name__ == "__main__":
    print("🔮 LSTM 모델 예측 시스템 (DB 연동)")
    print("=" * 60)
    
    # DB에서 모델 목록 조회
    print("\n📊 데이터베이스에서 모델 목록 조회 중...")
    models_df = get_available_models_from_db(show_stats=True)
    
    if models_df is None or models_df.empty:
        print("❌ 사용 가능한 모델이 없습니다.")
        exit()
    
    print(f"\n📋 사용 가능한 모델 목록 ({len(models_df)}개):")
    print("=" * 100)
    
    # 모델 목록 출력
    for idx, row in models_df.iterrows():
        print(f"\n{idx + 1}. [{row['model_id']}] {row['model_name']}")
        print(f"   타겟: {row['target_column']} | 시퀀스: {row['sequence_length']} | Epochs: {row['epochs']}")
        print(f"   총 실험: {row['total_experiments']}회", end='')
        
        if pd.notna(row['best_accuracy']):
            print(f" | 최고 정확도: {row['best_accuracy']:.2f}% (MAPE: {row['best_mape']:.2f}%)")
            if pd.notna(row['best_r2_score']):
                print(f"   R² Score: {row['best_r2_score']:.4f}", end='')
            print(f" | 마지막 실험: {row['last_experiment_date']}")
        else:
            print(" | 성공한 실험 없음")
    
    print("\n" + "=" * 100)
    
    # 모델 선택
    model_choice = input("\n모델 번호 또는 model_id 입력: ").strip()
    
    if model_choice.isdigit():
        choice_num = int(model_choice)
        if choice_num <= len(models_df):
            # 번호로 선택
            selected_model_id = models_df.iloc[choice_num - 1]['model_id']
        else:
            # ID로 선택
            selected_model_id = choice_num
    else:
        print("❌ 올바른 번호나 ID를 입력하세요.")
        exit()
    
    # 모델 로드
    print(f"\n{'='*60}")
    model_info = load_trained_model_from_db(model_id=selected_model_id, use_best_experiment=True)
    
    if model_info is None:
        print("❌ 모델 로드 실패. 프로그램을 종료합니다.")
        exit()
    
    config = model_info['config']
    
    # 사용 모드 선택
    print(f"\n{'='*60}")
    print("사용 가능한 모드:")
    print("1. DB에서 새 데이터 로드하여 예측")
    print("2. CSV 파일에서 데이터 로드하여 예측")
    print("3. 미래 예측 (재귀적)")
    print("4. 모델의 실험 이력 조회")
    
    mode = input("\n모드 선택 (1-4): ").strip()
    
    if mode == "1":
        # DB에서 데이터 로드
        print(f"\n📊 데이터 로드 설정:")
        print(f"   테이블: {config.get('tablename', 'lstm_input_1m')}")
        
        start_date = input("시작 날짜 (YYYY-MM-DD HH:MM:SS, Enter=전체): ").strip() or None
        end_date = input("종료 날짜 (YYYY-MM-DD HH:MM:SS, Enter=전체): ").strip() or None
        limit = input("최대 행 수 (Enter=제한없음): ").strip()
        limit = int(limit) if limit else None
        
        new_data = load_new_data_from_db(
            config.get('tablename', 'lstm_input_1m'),
            config['dateColumn'],
            config['studyColumns'],
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
        
        if new_data is not None:
            # 예측 수행
            result = predict_with_model(model_info, new_data)
            
            if result['status'] == 'success':
                # 결과 저장
                save_prediction_results(result)
                
                # 최근 5개 결과 출력
                print(f"\n📋 최근 5개 예측 결과:")
                for pred in result['predictions'][-5:]:
                    print(f"   {pred['date'][:19]}: 예측={pred['predicted_value']:.3f}", end='')
                    if pred['actual_value'] is not None:
                        print(f", 실제={pred['actual_value']:.3f}, 오차={pred['percentage_error']:.2f}%")
                    else:
                        print()
    
    elif mode == "2":
        # CSV 파일에서 로드
        csv_file = input("\nCSV 파일 경로: ").strip()
        
        try:
            new_data = pd.read_csv(csv_file)
            print(f"✅ CSV 파일 로드: {len(new_data)}행")
            
            # 예측 수행
            result = predict_with_model(model_info, new_data)
            
            if result['status'] == 'success':
                save_prediction_results(result)
                
        except Exception as e:
            print(f"❌ CSV 로드 실패: {str(e)}")
    
    elif mode == "3":
        # 미래 예측
        print(f"\n🔮 미래 예측 모드")
        
        # 초기 데이터 로드
        limit = config['r_seqLen'] + 100
        
        last_data = load_new_data_from_db(
            config.get('tablename', 'lstm_input_1m'),
            config['dateColumn'],
            config['studyColumns'],
            limit=limit
        )
        
        if last_data is not None:
            future_steps = int(input("예측할 미래 시점 수: ").strip() or "10")
            
            future_preds = predict_future(model_info, last_data, future_steps)
            
            if future_preds:
                print(f"\n📊 미래 예측 결과:")
                for i, pred in enumerate(future_preds, 1):
                    print(f"   Step {i}: {pred:.3f}")
    
    elif mode == "4":
        # 실험 이력 조회
        print(f"\n📊 모델의 실험 이력 조회")
        limit = int(input("조회할 실험 개수 (기본 10개): ").strip() or "10")
        
        experiments = get_experiments_by_model(selected_model_id, limit)
        
        if experiments is not None and not experiments.empty:
            print(f"\n📋 실험 이력 ({len(experiments)}개):")
            print(experiments.to_string(index=False))
        else:
            print("❌ 조회된 실험이 없습니다.")
    
    else:
        print("❌ 잘못된 모드 선택")
    
    print(f"\n{'='*60}")
    print("✅ 작업 완료!")