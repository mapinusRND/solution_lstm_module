# -*- coding: utf-8 -*-
"""
Title   : 전력 사용량 예측 LSTM (검증 기능 추가)
Author  : 주성중 / (주)맵인어스
Description: 
    - 원본 코드의 역스케일링 방식 유지
    - 학습 시와 동일한 날짜 필터만 추가
    - ✨ 실제 데이터 vs 예측 데이터 비교 기능 추가
Version : 2.8
Date    : 2025-10-23
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

# 환경 설정
ENV = os.getenv('FLASK_ENV', 'local')
if ENV == 'local':
    root = "D:/work/lstm"
else:
    root = "/app/webfiles/lstm"

model_path = os.path.abspath(root + "/saved_models")

PREDICTION_EPS_THRESHOLD = 0

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

def load_new_data(tablename, dateColumn, studyColumns, start_date=None, end_date=None, days_limit=7):
    try:
        engine = get_db_engine()
        
        if start_date is None and end_date is None:
            query = f"""
            SELECT {studyColumns},{dateColumn}
            FROM carbontwin.{tablename}
            WHERE {dateColumn} IS NOT NULL
            AND TO_CHAR({dateColumn}, 'MM-DD') NOT IN (
                '06-02', '06-13', '06-14', '06-15', '06-16', '06-17',
                '06-20', '06-21', '06-24', '06-25', '06-26', '06-28',
                '07-01', '07-08', '07-13', '07-14', '07-15', '07-16',
                '07-17', '07-18', '07-19', '07-21', '07-22'
            )
            ORDER BY {dateColumn} ASC
            """
        else:
            where_conditions = [f"{dateColumn} IS NOT NULL"]
            where_conditions.append(f"""TO_CHAR({dateColumn}, 'MM-DD') NOT IN (
                '06-02', '06-13', '06-14', '06-15', '06-16', '06-17',
                '06-20', '06-21', '06-24', '06-25', '06-26', '06-28',
                '07-01', '07-08', '07-13', '07-14', '07-15', '07-16',
                '07-17', '07-18', '07-19', '07-21', '07-22'
            )""")
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
        print(f"✅ 데이터 로드: {len(data)}행")
        
        if len(data) > 0 and dateColumn in data.columns:
            min_date = pd.to_datetime(data[dateColumn]).min()
            max_date = pd.to_datetime(data[dateColumn]).max()
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
        
        model = load_model(model_file, compile=False)
        model.compile(optimizer='adam', loss='mse')
        scaler = joblib.load(scaler_file)
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print(f"✅ 모델 로드 완료")
        print(f"   - 타겟: {config['targetColumn']}")
        print(f"   - 시퀀스: {config['r_seqLen']}")
        
        return model, scaler, config
        
    except Exception as e:
        print(f"❌ 모델 로드 오류: {str(e)}")
        return None, None, None

# ✨ 신규: 실제 데이터로 예측 검증하는 함수
def validate_with_actual_data(model, scaler, config, data, validation_days=7):
    """
    최근 N일치 데이터로 모델 예측 vs 실제값 비교
    
    Args:
        model: 학습된 모델
        scaler: 스케일러
        config: 모델 설정
        data: 전체 데이터
        validation_days: 검증할 일수 (기본 7일)
    
    Returns:
        dict: 검증 결과 (실제값, 예측값, 오차 등)
    """
    try:
        print(f"\n{'='*80}")
        print(f"🔍 모델 검증 시작 (최근 {validation_days}일 데이터)")
        print(f"{'='*80}")
        
        dateColumn = config['dateColumn']
        studyColumns = config['studyColumns']
        targetColumn = config['targetColumn']
        seq_len = int(config['r_seqLen'])
        
        study_columns_list = [col.strip() for col in studyColumns.split(',')]
        target_idx = study_columns_list.index(targetColumn)
        
        # 데이터 준비
        data_for_prediction = data[study_columns_list].astype(float)
        dates = pd.to_datetime(data[dateColumn])
        actual_values = data[targetColumn].values
        
        # 검증할 데이터 포인트 수 (15분 간격 * 96 * N일)
        validation_points = 96 * validation_days
        
        if len(data) < seq_len + validation_points:
            print(f"⚠️  데이터 부족: 최소 {seq_len + validation_points}개 필요, 현재 {len(data)}개")
            validation_points = len(data) - seq_len
            if validation_points <= 0:
                print("❌ 검증할 데이터 없음")
                return None
        
        # 정규화
        data_scaled = scaler.transform(data_for_prediction)
        
        # 검증 시작 인덱스 (최근 N일치)
        validation_start_idx = len(data) - validation_points
        
        print(f"\n📊 검증 설정:")
        print(f"   - 검증 기간: {dates.iloc[validation_start_idx]} ~ {dates.iloc[-1]}")
        print(f"   - 검증 포인트: {validation_points}개")
        print(f"   - 시퀀스 길이: {seq_len}")
        
        # 예측 수행
        predictions = []
        
        print(f"\n🔄 예측 진행 중...")
        
        for i in range(validation_start_idx, len(data)):
            # 시퀀스 추출 (i 이전 seq_len개)
            if i < seq_len:
                continue
                
            current_sequence = data_scaled[i - seq_len:i]
            
            # 모델 예측
            input_data = current_sequence.reshape(1, seq_len, len(study_columns_list))
            pred_scaled = model.predict(input_data, verbose=0)[0, 0]
            
            # 역스케일링 (원본과 동일한 방식)
            pred_original = pred_scaled * scaler.scale_[target_idx] + scaler.mean_[target_idx]
            pred_original = max(0, pred_original)
            
            predictions.append(pred_original)
            
            if (len(predictions)) % 100 == 0:
                print(f"   ⏳ {len(predictions)}/{validation_points} 완료")
        
        # 실제값과 비교
        actual_values_subset = actual_values[validation_start_idx:validation_start_idx + len(predictions)]
        dates_subset = dates.iloc[validation_start_idx:validation_start_idx + len(predictions)]
        
        # 오차 계산
        predictions_array = np.array(predictions)
        errors = predictions_array - actual_values_subset
        abs_errors = np.abs(errors)
        percentage_errors = np.abs(errors / (actual_values_subset + 1e-10)) * 100  # 0으로 나누기 방지
        
        # 통계
        mae = np.mean(abs_errors)
        rmse = np.sqrt(np.mean(errors ** 2))
        mape = np.mean(percentage_errors)
        
        print(f"\n{'='*80}")
        print(f"📈 검증 결과")
        print(f"{'='*80}")
        print(f"\n📊 전체 통계:")
        print(f"   MAE (평균 절대 오차):        {mae:.4f}")
        print(f"   RMSE (제곱근 평균 제곱 오차): {rmse:.4f}")
        print(f"   MAPE (평균 절대 백분율 오차): {mape:.2f}%")
        
        print(f"\n📊 실제값 범위:")
        print(f"   최소: {np.min(actual_values_subset):.4f}")
        print(f"   최대: {np.max(actual_values_subset):.4f}")
        print(f"   평균: {np.mean(actual_values_subset):.4f}")
        
        print(f"\n📊 예측값 범위:")
        print(f"   최소: {np.min(predictions_array):.4f}")
        print(f"   최대: {np.max(predictions_array):.4f}")
        print(f"   평균: {np.mean(predictions_array):.4f}")
        
        # 상세 비교표 출력 (샘플링)
        print(f"\n{'='*80}")
        print(f"📋 상세 비교 (매 시간마다 샘플링)")
        print(f"{'='*80}")
        print(f"{'날짜/시간':<22} {'실제값':>12} {'예측값':>12} {'오차':>12} {'오차율(%)':>12}")
        print(f"{'-'*80}")
        
        # 1시간마다 샘플링 (4개마다 = 15분 * 4 = 1시간)
        sample_indices = range(0, len(predictions), 4)
        
        for idx in sample_indices:
            if idx >= len(predictions):
                break
            
            date_str = dates_subset.iloc[idx].strftime('%Y-%m-%d %H:%M')
            actual = actual_values_subset[idx]
            pred = predictions_array[idx]
            error = errors[idx]
            error_pct = percentage_errors[idx]
            
            print(f"{date_str:<22} {actual:>12.4f} {pred:>12.4f} {error:>12.4f} {error_pct:>12.2f}")
        
        print(f"{'='*80}")
        
        # 일별 통계
        print(f"\n📅 일별 통계:")
        print(f"{'-'*80}")
        print(f"{'날짜':<12} {'실제 평균':>12} {'예측 평균':>12} {'MAE':>12} {'MAPE(%)':>12}")
        print(f"{'-'*80}")
        
        # 일별로 그룹화
        dates_only = dates_subset.dt.date
        unique_dates = dates_only.unique()
        
        for date in unique_dates:
            mask = dates_only == date
            daily_actual = actual_values_subset[mask]
            daily_pred = predictions_array[mask]
            daily_errors = np.abs(daily_pred - daily_actual)
            daily_pct_errors = np.abs((daily_pred - daily_actual) / (daily_actual + 1e-10)) * 100
            
            print(f"{str(date):<12} {np.mean(daily_actual):>12.4f} {np.mean(daily_pred):>12.4f} "
                  f"{np.mean(daily_errors):>12.4f} {np.mean(daily_pct_errors):>12.2f}")
        
        print(f"{'='*80}")
        
        # 결과 딕셔너리
        validation_result = {
            "validation_period": {
                "start_date": convert_to_serializable(dates_subset.iloc[0]),
                "end_date": convert_to_serializable(dates_subset.iloc[-1]),
                "days": validation_days,
                "points": len(predictions)
            },
            "statistics": {
                "mae": float(mae),
                "rmse": float(rmse),
                "mape": float(mape),
                "actual_min": float(np.min(actual_values_subset)),
                "actual_max": float(np.max(actual_values_subset)),
                "actual_mean": float(np.mean(actual_values_subset)),
                "predicted_min": float(np.min(predictions_array)),
                "predicted_max": float(np.max(predictions_array)),
                "predicted_mean": float(np.mean(predictions_array))
            },
            "comparison_data": [
                {
                    "date": convert_to_serializable(dates_subset.iloc[i]),
                    "actual": float(actual_values_subset[i]),
                    "predicted": float(predictions_array[i]),
                    "error": float(errors[i]),
                    "error_percentage": float(percentage_errors[i])
                }
                for i in range(len(predictions))
            ]
        }
        
        return validation_result
        
    except Exception as e:
        print(f"❌ 검증 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def predict_future_simple(model, scaler, config, data, future_steps=672):
    try:
        print(f"\n🔮 미래 예측 시작")
        print(f"   - 예측 스텝: {future_steps}개")
        
        dateColumn = config['dateColumn']
        studyColumns = config['studyColumns']
        targetColumn = config['targetColumn']
        seq_len = int(config['r_seqLen'])
        
        study_columns_list = [col.strip() for col in studyColumns.split(',')]
        target_idx = study_columns_list.index(targetColumn)
        
        # 마지막 날짜
        if dateColumn in data.columns:
            last_date = pd.to_datetime(data[dateColumn].iloc[-1])
        else:
            last_date = datetime.now()
        
        # 데이터 준비
        data_for_prediction = data[study_columns_list].astype(float)
        
        if len(data_for_prediction) < seq_len:
            raise ValueError(f"데이터 부족: {len(data_for_prediction)}개 (최소 {seq_len}개 필요)")
        
        # 정규화
        data_scaled = scaler.transform(data_for_prediction)
        
        # 시간 간격
        if dateColumn in data.columns and len(data) > 1:
            dates = pd.to_datetime(data[dateColumn])
            time_delta = (dates.iloc[-1] - dates.iloc[-2])
        else:
            time_delta = pd.Timedelta(minutes=15)
        
        # 현재 시퀀스
        current_sequence = data_scaled[-seq_len:].copy()
        
        # 결과 저장
        future_predictions = []
        future_dates = []
        
        print(f"   🔄 예측 진행 중...")
        
        # 예측 루프
        for step in range(future_steps):
            next_date = last_date + time_delta * (step + 1)
            
            # 모델 입력
            input_data = current_sequence.reshape(1, seq_len, len(study_columns_list))
            pred_scaled = model.predict(input_data, verbose=0)[0, 0]
            
            # 역스케일링
            pred_original = pred_scaled * scaler.scale_[target_idx] + scaler.mean_[target_idx]
            
            # 디버깅 (처음 5개)
            if step < 5:
                print(f"   [Step {step+1}] pred_scaled={pred_scaled:.6f}, pred_original={pred_original:.4f}")
            
            # 음수 방지
            pred_original = max(0, pred_original)
            
            future_predictions.append(pred_original)
            future_dates.append(next_date)
            
            # 다음 시퀀스 업데이트
            new_point = current_sequence[-1].copy()
            new_point_scaled = (pred_original - scaler.mean_[target_idx]) / scaler.scale_[target_idx]
            new_point[target_idx] = new_point_scaled
            
            # 슬라이딩 윈도우
            current_sequence = np.vstack([current_sequence[1:], new_point])
            
            if (step + 1) % 100 == 0:
                print(f"   ⏳ {step + 1}/{future_steps} 완료")
        
        print(f"✅ 예측 완료!")
        
        # 결과 포맷
        predictions_list = []
        for i, (pred_val, pred_date) in enumerate(zip(future_predictions, future_dates)):
            predictions_list.append({
                "date": convert_to_serializable(pred_date),
                "predicted_value": convert_to_serializable(pred_val),
                "is_reliable": True
            })
        
        # 통계
        print(f"\n📊 예측 결과:")
        print(f"   - 최소: {np.min(future_predictions):.2f}")
        print(f"   - 최대: {np.max(future_predictions):.2f}")
        print(f"   - 평균: {np.mean(future_predictions):.2f}")
        print(f"   - 표준편차: {np.std(future_predictions):.2f}")
        
        future_result = {
            "metadata": {
                "model_name": config.get('modelName', 'unknown'),
                "target_column": targetColumn,
                "sequence_length": seq_len,
                "prediction_steps": future_steps,
                "last_known_date": convert_to_serializable(last_date),
            },
            "predictions": predictions_list,
            "statistics": {
                "min_predicted": convert_to_serializable(np.min(future_predictions)),
                "max_predicted": convert_to_serializable(np.max(future_predictions)),
                "mean_predicted": convert_to_serializable(np.mean(future_predictions)),
                "std_predicted": convert_to_serializable(np.std(future_predictions))
            }
        }
        
        return future_result
        
    except Exception as e:
        print(f"❌ 예측 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def save_predictions_to_db(prediction_result, target_table="usage_generation_forecast"):
    if prediction_result is None:
        print("❌ 저장할 데이터 없음")
        return 0, 0
    
    try:
        engine = get_db_engine()
        predictions = prediction_result.get('predictions', [])
        
        if not predictions:
            print("❌ 예측 데이터 없음")
            return 0, 0
        
        print(f"\n💾 DB 저장 시작...")
        print(f"   - 테이블: carbontwin.{target_table}")
        print(f"   - 데이터: {len(predictions)}건")
        
        success_count = 0
        
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
                        (time_point, forecast_usage_kwh, reg_dt)
                    VALUES 
                        (:time_point, :forecast_value, CURRENT_TIMESTAMP)
                    """)
                    
                    conn.execute(insert_query, {
                        "time_point": time_point,
                        "forecast_value": forecast_value
                    })
                    
                    success_count += 1
                    
                    if success_count % 100 == 0:
                        print(f"   ⏳ {success_count}/{len(predictions)} 건")
                
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

def main(model_name=None, tablename=None, future_steps=672, save_to_db_flag=True, validation_days=3):
    print("=" * 70)
    print("⚡ 전력 사용량 예측 시스템 (검증 기능 추가)")
    print("=" * 70)
    
    model, scaler, config = load_trained_model(model_name)
    
    if model is None:
        return None
    
    print(f"\n📊 데이터 로드 중...")
    new_data = load_new_data(tablename, config['dateColumn'], config['studyColumns'])
    
    if new_data is None or new_data.empty:
        print("❌ 데이터 없음")
        return None
    
    # ✨ 검증 수행 (최근 3일 데이터로)
    validation_result = validate_with_actual_data(
        model, scaler, config, new_data, validation_days=validation_days
    )
    
    # 미래 예측
    print(f"\n⚡ 미래 예측")
    print(f"   - 예측: {future_steps}개 스텝 ({future_steps//96}일)")
    
    future_result = predict_future_simple(
        model, scaler, config, new_data, future_steps
    )
    
    if future_result and save_to_db_flag:
        success, fail = save_predictions_to_db(future_result)
        
        if success > 0:
            print(f"\n✅ {success}건 저장")
        if fail > 0:
            print(f"⚠️ {fail}건 실패")
    
    print(f"\n{'='*70}")
    print("🎉 완료!")
    print("="*70)
    
    return {
        "validation": validation_result,
        "future_prediction": future_result
    }

if __name__ == "__main__":
    try:
        model_name = "solar-hybrid-seq-2-test-20251017-test-no-add-usage_kwh"
        tablename = "lstm_input_15m_new"
        
        print("\n" + "=" * 80)
        print("⚡ 전력 사용량 예측 및 검증")
        print("=" * 80)
        
        # 예측 범위 설정
        future_steps = 192  # 2일
        validation_days = 7  # 최근 7일로 검증
        
        print(f"\n⚙️  설정:")
        print(f"   - 모델: {model_name}")
        print(f"   - 검증: 최근 {validation_days}일 데이터로 정확도 확인")
        print(f"   - 예측: {future_steps}개 스텝 ({future_steps//96}일)")

        result = main(
            model_name=model_name,
            tablename=tablename,
            future_steps=future_steps,
            save_to_db_flag=True,
            validation_days=validation_days
        )
        
        if result and result.get('validation'):
            val_stats = result['validation']['statistics']
            print(f"\n" + "="*80)
            print(f"📊 최종 검증 요약")
            print(f"="*80)
            print(f"   MAPE (평균 절대 백분율 오차): {val_stats['mape']:.2f}%")
            print(f"   MAE  (평균 절대 오차):        {val_stats['mae']:.4f}")
            print(f"   RMSE (제곱근 평균 제곱 오차): {val_stats['rmse']:.4f}")
            
            if val_stats['mape'] < 10:
                print(f"   ✅ 모델 성능: 우수 (MAPE < 10%)")
            elif val_stats['mape'] < 20:
                print(f"   ⚠️  모델 성능: 보통 (10% ≤ MAPE < 20%)")
            else:
                print(f"   ❌ 모델 성능: 개선 필요 (MAPE ≥ 20%)")
            print(f"="*80)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  중단")
    except Exception as e:
        print(f"\n❌ 오류: {str(e)}")
        import traceback
        traceback.print_exc()