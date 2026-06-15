import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import logging
import os
import sys
import json
from datetime import datetime
import shutil
from prophet import Prophet
import warnings
from google.cloud import bigquery

from dotenv import load_dotenv

warnings.filterwarnings("ignore")


load_dotenv()
# Setup de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BEST_MODEL_METRIC = os.getenv(
    "BEST_MODEL_METRIC", "validation_rmse"
)  # Lower is better for RMSE
BEST_MODEL_MODE = os.getenv("BEST_MODEL_MODE", "min")  # 'min' for RMSE, 'max' for R2
FORECAST_HORIZON = int(os.getenv("FORECAST_HORIZON", 30))  # Days to forecast


def load_training_data(engine):
    """Cargo los datos de entrenamiento desde la db"""

    logger.info("Cargando datos de entrenamiento desde la base de datos...")

    query = "SELECT * FROM taligent.iowa_liquor_sales where ordered_on >= '2024-01-01' and ordered_on < '2026-01-01' ORDER BY ordered_on"
    df = pd.read_sql(query, engine)
    logger.info(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas.")
    logger.info(f"Columnas: {df.columns.tolist()}")
    logger.info(f"Rango de fechas: {df['ordered_on'].min()} a {df['ordered_on'].max()}")

    return df


def load_training_data_bq():
    """Cargo los datos de entrenamiento desde la db"""
    client = bigquery.Client()
    logger.info("Cargando datos de entrenamiento desde la base de datos...")

    query = """
    SELECT 
        invoice_and_item_number,
        date,
        store_number,
        store_name,
        address,
        city,
        zip_code,
        store_location,
        county_number,
        county,
        category,
        category_name,
        vendor_number,
        vendor_name,
        item_number,
        item_description,
        pack,
        bottle_volume_ml,
        state_bottle_cost,
        state_bottle_retail,
        bottles_sold,
        sale_dollars,
        volume_sold_liters,
        volume_sold_gallons
    FROM `bigquery-public-data.iowa_liquor_sales.sales` 
    WHERE date >= '2025-01-01' AND date < '2026-01-01' 
    ORDER BY date
    """
    try:
        # 3. API request: Execute the query
        query_job = client.query(query)

        # 4. Wait for the query to finish and convert the results to a Pandas DataFrame
        df = query_job.to_dataframe()

        print("\n--- Query Successful! ---")
        print(df)
        logger.info(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas.")
        logger.info(f"Columnas: {df.columns.tolist()}")
        logger.info(f"Rango de fechas: {df['date'].min()} a {df['date'].max()}")
        return df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def prepare_time_series_data(df, horizon):
    """Preparar datos para forecasting de series temporales"""

    logger.info("Preparando datos para series temporales...")
    print(df.head())

    # Convertir fecha a datetime si no lo está
    df["date"] = pd.to_datetime(df["date"])

    # Agregar ventas por día (agregación temporal)
    df_daily = (
        df.groupby(["date", "county"])
        .agg(
            {
                "sale_dollars": "sum",
                "bottles_sold": "sum",
                "volume_sold_liters": "sum",
                "invoice_and_item_number": "count",
            }
        )
        .reset_index()
    )
    df_daily.columns = ["ds", "county", "y", "bottles", "liters", "transactions"]

    logger.info(f"Datos agregados por día: {df_daily.shape[0]} registros")
    logger.info(f"Rango de fechas: {df_daily['ds'].min()} a {df_daily['ds'].max()}")
    logger.info(f"Countries únicos: {df_daily['county'].nunique()}")

    # Feature engineering para series temporales
    df_daily["year"] = df_daily["ds"].dt.year
    df_daily["month"] = df_daily["ds"].dt.month
    df_daily["day"] = df_daily["ds"].dt.day
    df_daily["dayofweek"] = df_daily["ds"].dt.dayofweek
    df_daily["dayofyear"] = df_daily["ds"].dt.dayofyear
    df_daily["weekofyear"] = df_daily["ds"].dt.isocalendar().week.astype(int)
    df_daily["is_weekend"] = (df_daily["dayofweek"] >= 5).astype(int)
    df_daily["is_month_start"] = df_daily["ds"].dt.is_month_start.astype(int)
    df_daily["is_month_end"] = df_daily["ds"].dt.is_month_end.astype(int)
    df_daily["is_quarter_start"] = df_daily["ds"].dt.is_quarter_start.astype(int)
    df_daily["is_quarter_end"] = df_daily["ds"].dt.is_quarter_end.astype(int)

    # Lag features
    for lag in [1, 7, 14, 30]:
        df_daily[f"y_lag_{lag}"] = df_daily.groupby("county")["y"].shift(lag)

    # Rolling window features
    for window in [7, 14, 30]:
        df_daily[f"y_rolling_mean_{window}"] = df_daily.groupby("county")[
            "y"
        ].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        df_daily[f"y_rolling_std_{window}"] = df_daily.groupby("county")["y"].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )

    # Eliminar filas con NaN (por lag features)
    df_daily = df_daily.dropna()
    df_daily["target_y"] = df_daily.groupby("county")["y"].shift(-horizon)
    df_daily = df_daily[df_daily["target_y"].notna() & df_daily["y_lag_30"].notna()]
    logger.info(f"Datos después de feature engineering: {df_daily.shape[0]} registros")

    return df_daily


def split_time_series_data(df, test_size=0.2):
    """Split train/validation basado en tiempo (no aleatorio)"""

    # Ordenar por fecha
    df = df.sort_values("ds")

    # Calcular punto de corte basado en tiempo
    cutoff_date = df["ds"].quantile(1 - test_size)

    train_df = df[df["ds"] <= cutoff_date].copy()
    val_df = df[df["ds"] > cutoff_date].copy()

    logger.info(
        f"Train set: {train_df.shape[0]} registros ({train_df['ds'].min()} a {train_df['ds'].max()})"
    )
    logger.info(
        f"Validation set: {val_df.shape[0]} registros ({val_df['ds'].min()} a {val_df['ds'].max()})"
    )
    logger.info(f"Cutoff date: {cutoff_date}")

    # Stats del target
    logger.info("\nTarget (sales_dollars):")
    logger.info(
        f"  Train - mean: ${train_df['target_y'].mean():,.2f}, std: ${train_df['target_y'].std():,.2f}"
    )
    logger.info(
        f"  Val   - mean: ${val_df['target_y'].mean():,.2f}, std: ${val_df['target_y'].std():,.2f}"
    )

    return train_df, val_df


def train_prophet_model(train_df, val_df):
    """Entrenar modelo Prophet para forecasting"""

    logger.info("Entrenando modelo Prophet...")

    # Entrenar modelo Prophet para cada county
    models = {}
    forecasts = {}

    counties = train_df["county"].unique()
    logger.info(f"Entrenando modelos para {len(counties)} counties...")

    for county in counties:
        county_train = train_df[train_df["county"] == county][["ds", "y"]].copy()
        county_train.columns = ["ds", "y"]

        # Crear y entrenar modelo Prophet
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
            holidays_prior_scale=10,
        )

        try:
            model.fit(county_train)
            models[county] = model

            # Generar forecast para validation period
            future_dates = val_df[val_df["county"] == county]["ds"].unique()
            future = pd.DataFrame({"ds": future_dates})
            forecast = model.predict(future)
            forecasts[county] = forecast

        except Exception as e:
            logger.warning(f"Error entrenando modelo para county {county}: {e}")

    logger.info(f"Modelos Prophet entrenados: {len(models)}/{len(counties)}")
    return models, forecasts


def train_xgboost_forecaster(train_df, val_df):
    """Entrenar XGBoost para forecasting con features temporales"""

    logger.info("Entrenando XGBoost forecaster...")

    # Features para el modelo
    feature_cols = [
        "year",
        "month",
        "day",
        "dayofweek",
        "dayofyear",
        "weekofyear",
        "is_weekend",
        "is_month_start",
        "is_month_end",
        "is_quarter_start",
        "is_quarter_end",
        "y_lag_1",
        "y_lag_7",
        "y_lag_14",
        "y_lag_30",
        "y_rolling_mean_7",
        "y_rolling_std_7",
        "y_rolling_mean_14",
        "y_rolling_std_14",
        "y_rolling_mean_30",
        "y_rolling_std_30",
    ]

    # One-hot encode county
    train_df_encoded = pd.get_dummies(train_df, columns=["county"], prefix="county")
    val_df_encoded = pd.get_dummies(val_df, columns=["county"], prefix="county")

    # Asegurar que ambas columnas tengan las mismas columnas
    train_cols = set(train_df_encoded.columns)
    val_cols = set(val_df_encoded.columns)
    missing_cols = train_cols - val_cols
    for col in missing_cols:
        val_df_encoded[col] = 0
    extra_cols = val_cols - train_cols
    for col in extra_cols:
        val_df_encoded.drop(col, axis=1, inplace=True)

    # Ordenar columnas
    val_df_encoded = val_df_encoded[train_df_encoded.columns]

    # Separar features y target
    X_train = train_df_encoded[
        feature_cols + [c for c in train_df_encoded.columns if c.startswith("county_")]
    ]
    y_train = train_df_encoded["target_y"]
    X_val = val_df_encoded[
        feature_cols + [c for c in val_df_encoded.columns if c.startswith("county_")]
    ]

    # Entrenar modelo
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
    )

    model.fit(X_train, y_train)

    # Predicciones
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    logger.info("XGBoost forecaster entrenado")
    logger.info(f"Features usadas: {X_train.shape[1]}")

    return model, y_train_pred, y_val_pred, X_train.columns.tolist()


def evaluate_forecast_model(val_df, forecasts, model_type="prophet"):
    """Evaluar modelo de forecasting"""

    logger.info("EVALUACIÓN DEL MODELO DE FORECASTING")

    if model_type == "prophet":
        # Evaluar Prophet forecasts
        all_predictions = []
        all_actuals = []

        for county, forecast in forecasts.items():
            county_val = val_df[val_df["county"] == county]
            if len(county_val) > 0:
                # Merge forecast with actuals
                merged = pd.merge(
                    forecast[["ds", "yhat"]],
                    county_val[["ds", "target_y"]],
                    on="ds",
                    how="inner",
                )
                all_predictions.extend(merged["yhat"].values)
                all_actuals.extend(merged["target_y"].values)

        y_true = np.array(all_actuals)
        y_pred = np.array(all_predictions)

    elif model_type == "xgboost":
        # Evaluar XGBoost predictions (passed separately)
        return None  # Handled in train_xgboost_forecaster

    # Calcular métricas
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    # Filter out near-zero values for MAPE to avoid extreme percentages
    mask = np.abs(y_true) > 0.01
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan

    # Métricas adicionales para forecasting
    smape = (
        np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100
    )

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "smape": smape,
    }

    logger.info("Métricas de forecasting:")
    logger.info(f"  RMSE: ${rmse:,.2f}")
    logger.info(f"  MAE: ${mae:,.2f}")
    logger.info(f"  MAPE: {mape:.2f}%")
    logger.info(f"  SMAPE: {smape:.2f}%")

    return metrics


def evaluate_xgboost_forecast(y_train, y_train_pred, y_val, y_val_pred):
    """Evaluar XGBoost forecaster"""

    logger.info("EVALUACIÓN DEL MODELO XGBOOST FORECASTER")

    def calculate_metrics(y_true, y_pred, dataset_name):
        rmse = np.sqrt(root_mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        # Filter out near-zero values for MAPE to avoid extreme percentages
        mask = np.abs(y_true) > 0.01
        if np.sum(mask) > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = np.nan
        smape = (
            np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
            * 100
        )

        metrics = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mape": mape,
            "smape": smape,
        }

        logger.info(f"{dataset_name.upper()} SET:")
        logger.info(f"  RMSE: ${rmse:,.2f}")
        logger.info(f"  MAE: ${mae:,.2f}")
        logger.info(f"  R²: {r2:.4f}")
        logger.info(f"  MAPE: {mape:.2f}%")
        logger.info(f"  SMAPE: {smape:.2f}%")

        return metrics

    train_metrics = calculate_metrics(y_train, y_train_pred, "train")
    val_metrics = calculate_metrics(y_val, y_val_pred, "validation")

    # Overfitting analysis
    r2_diff = train_metrics["r2"] - val_metrics["r2"]
    logger.info("OVERFITTING ANALYSIS:")
    logger.info(f"  R² difference (train - val): {r2_diff:.4f}")

    return {
        "train": train_metrics,
        "validation": val_metrics,
        "overfitting_score": r2_diff,
    }


def save_forecast_model(model, metrics, model_type, output_dir="models"):
    """Guarda modelo de forecasting y metadata"""

    logger.info("Guardando modelo de forecasting y metadata...")

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Guardar modelo
    if model_type == "xgboost":
        model_filename = f"xgboost_forecaster_{timestamp}.pkl"
        model_path = os.path.join(output_dir, model_filename)
        joblib.dump(model, model_path, compress=3)
    elif model_type == "prophet":
        # Prophet models are dictionaries of county-specific models
        model_filename = f"prophet_models_{timestamp}.pkl"
        model_path = os.path.join(output_dir, model_filename)
        joblib.dump(model, model_path, compress=3)

    logger.info(f"Modelo guardado en: {model_path}")

    # Guardar metadata
    metadata = {
        "timestamp": timestamp,
        "model_type": model_type,
        "task": "time_series_forecasting",
        "train_metrics": metrics["train"],
        "validation_metrics": metrics["validation"],
        "overfitting_score": metrics["overfitting_score"],
        "forecast_horizon": FORECAST_HORIZON,
    }
    metadata_filename = f"forecast_metadata_{timestamp}.json"
    metadata_path = os.path.join(output_dir, metadata_filename)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata guardada en: {metadata_path}")

    # Crear symlink "latest"
    latest_model_path = os.path.join(output_dir, "best_forecast_model.pkl")
    latest_metadata_path = os.path.join(output_dir, "best_forecast_metadata.json")

    shutil.copy2(model_path, latest_model_path)
    shutil.copy2(metadata_path, latest_metadata_path)
    logger.info(f"Latest model actualizado: {latest_model_path}")

    return model_path


def generate_forecast_report(metrics, model_type, output_dir="results"):
    """Generar reporte de forecasting"""

    logger.info("Generando reporte de forecasting...")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"forecast_report_{timestamp}.txt")

    with open(report_path, "w") as f:
        f.write("IOWA LIQUOR SALES FORECASTING - TRAINING REPORT\n")

        f.write(f"\nFecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Timestamp: {timestamp}\n\n")

        f.write(f"Modelo seleccionado: {model_type.upper()}\n")
        f.write("\nJustificación\n")
        if model_type == "prophet":
            f.write(
                "Prophet fue seleccionado por su capacidad para manejar series temporales\n"
            )
            f.write(
                "con estacionalidad múltiple, patrones de holidays y tendencias cambiantes.\n"
            )
            f.write(
                "Es especialmente adecuado para datos de ventas con patrones semanales y anuales.\n"
            )
        elif model_type == "xgboost":
            f.write(
                "XGBoost fue seleccionado por su capacidad para capturar relaciones complejas\n"
            )
            f.write(
                "entre features temporales, lag features y variables categóricas como county.\n"
            )
            f.write(
                "Es eficiente y maneja bien grandes volúmenes de datos con features engineered.\n"
            )

        f.write("\nMétricas de evaluación:\n")
        f.write("-" * 70 + "\n")
        f.write("TRAIN SET:\n")
        f.write(f"  RMSE:        ${metrics['train']['rmse']:>12,.2f}\n")
        f.write(f"  MAE:         ${metrics['train']['mae']:>12,.2f}\n")
        f.write(f"  R²:          {metrics['train']['r2']:>13.4f}\n")
        f.write(f"  MAPE:        {metrics['train']['mape']:>12.2f}%\n")
        f.write(f"  SMAPE:       {metrics['train']['smape']:>12.2f}%\n")
        f.write("\nVALIDATION SET:\n")
        f.write(f"  RMSE:        ${metrics['validation']['rmse']:>12,.2f}\n")
        f.write(f"  MAE:         ${metrics['validation']['mae']:>12,.2f}\n")
        f.write(f"  R²:          {metrics['validation']['r2']:>13.4f}\n")
        f.write(f"  MAPE:        {metrics['validation']['mape']:>12.2f}%\n")
        f.write(f"  SMAPE:       {metrics['validation']['smape']:>12.2f}%\n")

        f.write("\nInterpretación de resultados:\n")
        f.write("-" * 70 + "\n")
        r2_pct = metrics["validation"]["r2"] * 100
        f.write(
            f"El modelo explica aproximadamente {r2_pct:.2f}% de la varianza en las ventas\n"
        )
        f.write("de licor en el set de validación.\n\n")
        f.write(
            f"Error promedio absoluto (MAE) de ${metrics['validation']['mae']:,.2f} por predicción\n"
        )
        f.write(
            f"Error porcentual medio (MAPE) de {metrics['validation']['mape']:.2f}%\n"
        )
        f.write(
            f"Error porcentual simétrico (SMAPE) de {metrics['validation']['smape']:.2f}%\n\n"
        )

        overfitting = metrics["overfitting_score"]
        if overfitting > 0.1:
            f.write(
                f"Se detecta un posible overfitting (R² train - R² val = {overfitting:.4f}).\n"
            )
            f.write("Considerar técnicas de regularización o más datos históricos.\n")
        else:
            f.write(
                f"No se detecta un overfitting significativo (R² train - R² val = {overfitting:.4f}).\n"
            )
            f.write("El modelo parece generalizar bien al set de validación.\n")

        f.write(f"\nHorizonte de forecast: {FORECAST_HORIZON} días\n")

    logger.info(f"Reporte generado en: {report_path}")
    return report_path


def save_best_model_artifacts(model, metrics, best_params, output_dir="models"):
    """Guardar artefactos del mejor modelo para consumo por scoring"""
    logger.info("Guardando artefactos del mejor modelo para scoring...")

    os.makedirs(output_dir, exist_ok=True)

    # Guardar modelo con nombre estándar para scoring
    model_path = os.path.join(output_dir, "best_model.pkl")
    joblib.dump(model, model_path, compress=3)

    # Guardar metadata con información del run de MLflow
    metadata = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "model_type": "XGBoostRegressor",
        "best_params": best_params,
        "train_metrics": metrics["train"],
        "validation_metrics": metrics["validation"],
        "overfitting_score": metrics["overfitting_score"],
        "best_model_metric": BEST_MODEL_METRIC,
        "best_model_metric_value": metrics["validation"][
            BEST_MODEL_METRIC.replace("validation_", "")
        ],
    }

    metadata_path = os.path.join(output_dir, "best_model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Mejor modelo guardado en: {model_path}")
    logger.info(f"Metadata del mejor modelo guardada en: {metadata_path}")

    return model_path, metadata_path


def main():
    """Función principal para ejecutar el proceso de forecasting."""

    try:
        logger.info("Iniciando proceso de forecasting...")

        df = load_training_data_bq()

        # Paso 3: Preparar datos para series temporales
        logger.info("Preparando datos para series temporales...")
        df_ts = prepare_time_series_data(df, FORECAST_HORIZON)

        # Paso 4: Split train/validation basado en tiempo
        test_size = 0.2
        logger.info("Dividiendo datos en train y validation (time-based)...")
        train_df, val_df = split_time_series_data(df_ts, test_size=test_size)

        # Seleccionar modelo de forecasting
        model_type = os.getenv("FORECAST_MODEL", "xgboost")  # 'prophet' or 'xgboost'
        logger.info(f"Modelo seleccionado: {model_type}")

        if model_type == "prophet":
            logger.info("Entrenando modelos Prophet por county...")
            prophet_models, prophet_forecasts = train_prophet_model(train_df, val_df)
            metrics = evaluate_forecast_model(val_df, prophet_forecasts, "prophet")
            model_path = save_forecast_model(prophet_models, metrics, "prophet")

        elif model_type == "xgboost":
            logger.info("Entrenando XGBoost forecaster...")
            xgb_model, y_train_pred, y_val_pred, feature_list = (
                train_xgboost_forecaster(train_df, val_df)
            )
            metrics = evaluate_xgboost_forecast(
                train_df["y"], y_train_pred, val_df["y"], y_val_pred
            )
            model_path = save_forecast_model(xgb_model, metrics, "xgboost")

        # Paso 6: Generar reporte de forecasting
        report_path = generate_forecast_report(metrics, model_type)

        logger.info("\n" + "=" * 70)
        logger.info("FORECASTING PIPELINE COMPLETADO EXITOSAMENTE")
        logger.info("=" * 70)
        logger.info(f"\nModelo guardado en: {model_path}")
        logger.info(f"Reporte generado en: {report_path}")
        logger.info(f"\nValidation R²: {metrics['validation']['r2']:.4f}")
        logger.info(f"Validation RMSE: ${metrics['validation']['rmse']:,.2f}")
        logger.info(f"Validation MAPE: {metrics['validation']['mape']:.2f}%")
        logger.info(f"Forecast Horizon: {FORECAST_HORIZON} días")

        return True

    except Exception as e:
        logger.error(f"\nError en el proceso de forecasting: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
