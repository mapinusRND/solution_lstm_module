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
    """LSTM 모델을 사용한 미래 전력 사용량 예측 함수 (안정화 최소화)"""
    
    def calculate_workday_holiday_patterns(data_for_prediction, dates, targetColumn):
        """과거 데이터에서 평일/휴일 패턴을 자동으로 학습"""
        print(f"   🔍 평일/휴일 패턴 학습 중...")
        
        target_values = data_for_prediction[targetColumn].values
        weekday_mask = dates.dt.weekday < 5
        weekend_mask = dates.dt.weekday >= 5
        
        weekday_values = target_values[weekday_mask]
        weekend_values = target_values[weekend_mask]
        
        patterns = {
            "workday": {
                "mean": float(np.mean(weekday_values)),
                "std": float(np.std(weekday_values)),
                "median": float(np.median(weekday_values)),
                "min": float(np.min(weekday_values)),
                "max": float(np.max(weekday_values)),
                "q25": float(np.percentile(weekday_values, 25)),
                "q75": float(np.percentile(weekday_values, 75)),
                "zero_ratio": float(np.sum(weekday_values == 0) / len(weekday_values)),
                "count": len(weekday_values)
            },
            "holiday": {
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
    
    def adaptive_stabilization(pred_original, next_date, patterns):
        """
        ✅ 예측값을 거의 그대로 사용 (극단적 이상치만 제거)
        
        목표:
        - 평일: 모델 예측 그대로 (101 kWh 정도)
        - 휴일: 모델 예측 그대로 (0 kWh 근처)
        - 명백한 오류만 제거 (음수, 비현실적 대량값)
        """
        day_of_week = next_date.weekday()
        is_workday = day_of_week < 5
        
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
        
        # ====================================================================
        # ✅ 매우 느슨한 범위 (5σ) - 99.9999% 범위
        # ====================================================================
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
        
        # ❌ 평균 회귀 제거
        # ❌ IQR 제한 제거
        # ❌ 랜덤 대체 제거
        # → 모델 예측을 최대한 그대로 사용!
        
        pred_original = max(0, pred_original)
        
        return pred_original, stabilization_applied, stabilization_reason, weekday_name, day_type, icon
    
    def update_time_features(next_row, next_date, study_columns_list):
        """시간 관련 특성 동적 업데이트"""
        day_of_week = next_date.weekday()
        
        if 'week_code' in study_columns_list:
            idx = study_columns_list.index('week_code')
            next_row[idx] = min(day_of_week + 1, 6)
        
        if 'is_weekend' in study_columns_list:
            idx = study_columns_list.index('is_weekend')
            next_row[idx] = 1 if day_of_week >= 5 else 0
        
        if 'is_workday' in study_columns_list:
            idx = study_columns_list.index('is_workday')
            next_row[idx] = 1 if day_of_week < 5 else 0
        
        if 'day_sin' in study_columns_list:
            idx = study_columns_list.index('day_sin')
            next_row[idx] = np.sin(2 * np.pi * day_of_week / 7)
        
        if 'day_cos' in study_columns_list:
            idx = study_columns_list.index('day_cos')
            next_row[idx] = np.cos(2 * np.pi * day_of_week / 7)
        
        return next_row
    
    # 메인 예측 로직
    try:
        print(f"\n{'='*80}")
        print(f"🔮 평일/휴일 기반 예측 ({future_steps}개 스텝 = {future_steps//96}일)")
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
        
        patterns, weekday_details = calculate_workday_holiday_patterns(
            data_for_prediction, dates, targetColumn
        )
        
        if historical_mean is None:
            historical_mean = data_for_prediction[targetColumn].mean()
        if historical_std is None:
            historical_std = data_for_prediction[targetColumn].std()
        
        print(f"\n   📊 학습된 패턴:")
        print(f"      🏢 평일 (월~금):")
        print(f"         - 평균: {patterns['workday']['mean']:6.2f} kWh (±{patterns['workday']['std']:5.2f})")
        print(f"         - 범위: [{patterns['workday']['min']:.2f}, {patterns['workday']['max']:.2f}]")
        print(f"         - 0값 비율: {patterns['workday']['zero_ratio']*100:4.1f}%")
        print(f"         - 데이터 수: {patterns['workday']['count']:,}개")
        
        print(f"\n      🏖️ 휴일 (토, 일):")
        print(f"         - 평균: {patterns['holiday']['mean']:6.2f} kWh (±{patterns['holiday']['std']:5.2f})")
        print(f"         - 범위: [{patterns['holiday']['min']:.2f}, {patterns['holiday']['max']:.2f}]")
        print(f"         - 0값 비율: {patterns['holiday']['zero_ratio']*100:4.1f}%")
        print(f"         - 데이터 수: {patterns['holiday']['count']:,}개")
        
        print(f"\n   📅 요일별 상세:")
        for day_idx in range(7):
            if day_idx in weekday_details:
                detail = weekday_details[day_idx]
                icon = "🏢" if detail["is_workday"] else "🏖️"
                print(f"      {icon} {detail['name']}요일: {detail['mean']:6.2f} kWh "
                      f"(±{detail['std']:5.2f}) | 0값: {detail['zero_ratio']*100:4.1f}%")
        
        print(f"\n   🔄 역정규화: 전체 피처 벡터 방식")
        print(f"   ✅ 안정화: 5σ 범위 (극단적 이상치만 제거)")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data_scaled = scaler.transform(data_for_prediction)
        
        current_sequence = data_scaled[-seq_len:, :].copy()
        
        future_predictions = []
        future_dates = []
        stabilization_log = []
        max_log = 10
        
        for step in range(future_steps):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                X = current_sequence.reshape(1, seq_len, -1)
                pred_scaled = model.predict(X, verbose=0)[0, 0]
                
                full_scaled = current_sequence[-1].copy()
                full_scaled[target_idx] = pred_scaled
                full_original = scaler.inverse_transform(full_scaled.reshape(1, -1))[0]
                pred_original = float(full_original[target_idx])
            
            next_date = last_date + timedelta(minutes=15 * (step + 1))
            
            pred_original, stabilized, reason, weekday_name, day_type, icon = adaptive_stabilization(
                pred_original, next_date, patterns
            )
            
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
            
            future_predictions.append(pred_original)
            future_dates.append(next_date)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                next_row = data_for_prediction.iloc[-1].copy().values
                next_row[target_idx] = pred_original
                next_row = update_time_features(next_row, next_date, study_columns_list)
                next_row_scaled = scaler.transform(next_row.reshape(1, -1))[0].astype(np.float32)
                
                if np.any(np.isnan(next_row_scaled)) or np.any(np.isinf(next_row_scaled)):
                    next_row_scaled = np.mean(current_sequence[-10:], axis=0)
            
            current_sequence = np.vstack([current_sequence[1:], next_row_scaled.reshape(1, -1)])
            
            if (step + 1) % 96 == 0:
                print(f"   ⏳ {step + 1}/{future_steps} 완료 ({(step+1)//96}일)")
        
        if stabilization_log:
            print(f"\n   ⚠️  안정화 적용 사례 (총 {len(stabilization_log)}건):")
            for log in stabilization_log[:5]:
                print(f"      {log['icon']} {log['date']} ({log['weekday']}, {log['type']}): "
                      f"{log['value']:.2f} kWh - {log['reason']}")
            if len(stabilization_log) > 5:
                print(f"      ... 외 {len(stabilization_log) - 5}건")
        
        future_predictions = np.array(future_predictions)
        
        print(f"\n📊 예측 결과:")
        print(f"   - 최소: {np.min(future_predictions):.2f} kWh")
        print(f"   - 최대: {np.max(future_predictions):.2f} kWh")
        print(f"   - 평균: {np.mean(future_predictions):.2f} kWh")
        print(f"   - 표준편차: {np.std(future_predictions):.2f} kWh")
        
        workday_predictions = []
        holiday_predictions = []
        
        for pred_val, pred_date in zip(future_predictions, future_dates):
            if pred_date.weekday() < 5:
                workday_predictions.append(pred_val)
            else:
                holiday_predictions.append(pred_val)
        
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
        # 현재 usage-kwh-model-4 모델이 가장 유사한 형태
        model_name = "usage-kwh-model-4"
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