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

def validate_with_actual_data(model, scaler, config, data, validation_days=2):
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
    미래 예측 - 개선된 역정규화 방식
    """
    try:
        print(f"\n{'='*80}")
        print(f"🔮 안정적 미래 예측 ({future_steps}개 스텝 = {future_steps//96}일)")
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
        last_date = dates.iloc[-1]
        
        # 평균/표준편차 기본값 설정
        if historical_mean is None:
            historical_mean = data_for_prediction[targetColumn].mean()
        if historical_std is None:
            historical_std = data_for_prediction[targetColumn].std()
        
        print(f"   📊 역사적 통계:")
        print(f"      - 평균: {historical_mean:.2f}")
        print(f"      - 표준편차: {historical_std:.2f}")
        print(f"   🔄 역정규화: 전체 피처 벡터 방식")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data_scaled = scaler.transform(data_for_prediction)
        
        current_sequence = data_scaled[-seq_len:, :].copy()
        
        future_predictions = []
        future_dates = []
        
        warning_count = 0
        max_warnings = 5
        
        for step in range(future_steps):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                X = current_sequence.reshape(1, seq_len, -1)
                pred_scaled = model.predict(X, verbose=0)[0, 0]
                
                # ✅ 개선된 역정규화: 전체 피처 벡터 방식
                full_scaled = current_sequence[-1].copy()  # 마지막 스텝의 정규화 공간 벡터 복사
                full_scaled[target_idx] = pred_scaled  # 타겟 위치에 예측값 삽입
                full_original = scaler.inverse_transform(full_scaled.reshape(1, -1))[0]
                pred_original = float(full_original[target_idx])
            
            # ✅ 평균 회귀 적용 (안정화)
            deviation = abs(pred_original - historical_mean)
            if deviation > 3 * historical_std:
                alpha = 0.7
                pred_original = alpha * historical_mean + (1 - alpha) * pred_original
                if warning_count < max_warnings:
                    print(f"   📉 Step {step}: 평균 회귀 적용 ({pred_original:.2f})")
                    warning_count += 1
            
            # ✅ 안전 범위 제한
            safe_min = max(0, historical_mean - 3 * historical_std)
            safe_max = historical_mean + 3 * historical_std
            pred_original = np.clip(pred_original, safe_min, safe_max)
            
            future_predictions.append(pred_original)
            next_date = last_date + timedelta(minutes=15 * (step + 1))
            future_dates.append(next_date)
            
            # ✅ 시퀀스 업데이트
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                next_row = data_for_prediction.iloc[-1].copy().values
                next_row[target_idx] = pred_original
                
                # 시간 특성 업데이트
                hour = next_date.hour
                is_workday = 1 if 7 <= hour <= 20 else 0
                next_row[0] = is_workday
                
                next_row_scaled = scaler.transform(next_row.reshape(1, -1))[0].astype(np.float32)
                
                if np.any(np.isnan(next_row_scaled)) or np.any(np.isinf(next_row_scaled)):
                    next_row_scaled = np.mean(current_sequence[-10:], axis=0)
            
            current_sequence = np.vstack([current_sequence[1:], next_row_scaled.reshape(1, -1)])
            
            if (step + 1) % 96 == 0:
                print(f"   ⏳ {step + 1}/{future_steps} 완료 ({(step+1)//96}일)")
        
        if warning_count >= max_warnings:
            print(f"   ⚠️  총 {warning_count}번의 평균 회귀 적용됨")
        
        future_predictions = np.array(future_predictions)
        
        print(f"\n📊 예측 결과:")
        print(f"   - 최소: {np.min(future_predictions):.2f}")
        print(f"   - 최대: {np.max(future_predictions):.2f}")
        print(f"   - 평균: {np.mean(future_predictions):.2f}")
        print(f"   - 표준편차: {np.std(future_predictions):.2f}")
        print(f"   - 역사적 평균과 차이: {abs(np.mean(future_predictions) - historical_mean):.2f}")
        
        predictions_list = []
        for pred_val, pred_date in zip(future_predictions, future_dates):
            predictions_list.append({
                "date": convert_to_serializable(pred_date),
                "predicted_value": convert_to_serializable(pred_val)
            })
        
        return {
            "metadata": {
                "model_name": config.get('modelName', 'unknown'),
                "target_column": targetColumn,
                "prediction_steps": future_steps,
                "last_known_date": convert_to_serializable(last_date),
                "method": "전체 피처 벡터 역정규화 + 평균 회귀",
                "historical_mean": historical_mean,
                "historical_std": historical_std
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

def main(model_name, tablename, future_steps=672, save_to_db_flag=True, validation_days=2):
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
        model_name = "usage-kwh-model-5"
        tablename = "lstm_input_15m_new"
        
        print("\n" + "=" * 80)
        print("⚡ 개선된 역정규화 방식 적용")
        print("=" * 80)
        
        result = main(
            model_name=model_name,
            tablename=tablename,
            future_steps=672,
            save_to_db_flag=True,
            validation_days=2
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