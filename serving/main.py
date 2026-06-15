import os
import logging
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import google.cloud.storage as storage
import google.cloud.bigquery as bigquery


from typing import Any, Optional

# Configure standard logger to work with Uvicorn/Cloud Run
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.error")
app = FastAPI()

# Global variables to hold our model and metadata
MODEL: Optional[Any] = None
METADATA: Optional[Any] = None


def download_model(gcs_path, local_path="/tmp/model.pkl"):
    gcs_path = gcs_path.replace("gs://", "")
    bucket_name, blob_path = gcs_path.split("/", 1)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    bucket.blob(blob_path).download_to_filename(local_path)
    return local_path


def load_model():
    global MODEL, METADATA
    gcs_path = os.environ.get("GCS_MODEL_PATH")
    if not gcs_path:
        logger.warning("GCS_MODEL_PATH not found. Falling back to local file path.")
        local_path = "./best_forecast_model.pkl"
    else:
        local_path = download_model(gcs_path)
    local_path = "./models/best_forecast_model.pkl"

    # Corrected: use joblib to safely open the compressed .pkl file
    MODEL = joblib.load(local_path)
    METADATA = None  # You aren't storing separate metadata in this file

    print("Model successfully decompressed and loaded from GCS via joblib.")


# Load model when container starts
load_model()


class ForecastRequest(BaseModel):
    target_date: str = Field(
        ..., description="The date to forecast for, format YYYY-MM-DD"
    )
    county: str = Field(
        ..., description="The target county/country name matching BQ data"
    )


@app.post("/predict")
def predict(payload: ForecastRequest):
    features_df = get_data_forecast_request(payload)
    logger.info("---------- PROCESSED FEATURE DATAFRAME FOR MODEL INPUT ----------")
    # This shows all columns, including your generated lag/rolling metrics
    logger.info(f"\n{features_df.to_string(index=False)}")
    logger.info("-----------------------------------------------------------------")

    # 2. Extract only the row matching the specific target date to forecast
    target_row_df = features_df[
        features_df["ds"].dt.strftime("%Y-%m-%d") == payload.target_date
    ]

    if target_row_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Could not calculate feature matrices for specific target date: {payload.target_date}. Check historical row distribution.",
        )

    # 3. Log the fully processed feature DataFrame matrix to the console
    logger.info("---------- PROCESSED FEATURE DATAFRAME FOR MODEL INPUT ----------")
    # This shows all columns, including your generated lag/rolling metrics
    logger.info(f"\n{target_row_df.to_string(index=False)}")
    logger.info("-----------------------------------------------------------------")

    # 4. One-hot encode counties and sort feature columns for model matching
    prepared_df = prepare_features(target_row_df)
    if MODEL is None:
        logger.error("Prediction failed: Model artifact was never successfully loaded.")
        raise HTTPException(
            status_code=500,
            detail="Model is not initialized. Please check server deployment logs.",
        )
    # 5. Run inference
    predictions = MODEL.predict(prepared_df)

    return {
        "target_date": payload.target_date,
        "county": payload.county,
        "forecast": predictions.tolist(),
    }


@app.get("/healthz")
def health_check():
    return {"status": "healthy"}


def prepare_features(payload_df: pd.DataFrame) -> pd.DataFrame:
    """Prepara los features para el modelo matching structural shape"""
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
        "county_ADAIR",
        "county_ADAMS",
        "county_ALLAMAKEE",
        "county_APPANOOSE",
        "county_AUDUBON",
        "county_BENTON",
        "county_BLACK HAWK",
        "county_BOONE",
        "county_BREMER",
        "county_BUCHANAN",
        "county_BUENA VISTA",
        "county_BUTLER",
        "county_CALHOUN",
        "county_CARROLL",
        "county_CASS",
        "county_CEDAR",
        "county_CERRO GORDO",
        "county_CHEROKEE",
        "county_CHICKASAW",
        "county_CLARKE",
        "county_CLAY",
        "county_CLAYTON",
        "county_CLINTON",
        "county_CRAWFORD",
        "county_DALLAS",
        "county_DAVIS",
        "county_DECATUR",
        "county_DELAWARE",
        "county_DES MOINES",
        "county_DICKINSON",
        "county_DUBUQUE",
        "county_EMMET",
        "county_FAYETTE",
        "county_FLOYD",
        "county_FRANKLIN",
        "county_FREMONT",
        "county_GREENE",
        "county_GRUNDY",
        "county_GUTHRIE",
        "county_HAMILTON",
        "county_HANCOCK",
        "county_HARDIN",
        "county_HARRISON",
        "county_HENRY",
        "county_HOWARD",
        "county_HUMBOLDT",
        "county_IDA",
        "county_IOWA",
        "county_JACKSON",
        "county_JASPER",
        "county_JEFFERSON",
        "county_JOHNSON",
        "county_JONES",
        "county_KEOKUK",
        "county_KOSSUTH",
        "county_LEE",
        "county_LINN",
        "county_LOUISA",
        "county_LUCAS",
        "county_LYON",
        "county_MADISON",
        "county_MAHASKA",
        "county_MARION",
        "county_MARSHALL",
        "county_MILLS",
        "county_MITCHELL",
        "county_MONONA",
        "county_MONROE",
        "county_MONTGOMERY",
        "county_MUSCATINE",
        "county_O'BRIEN",
        "county_OSCEOLA",
        "county_PAGE",
        "county_PALO ALTO",
        "county_PLYMOUTH",
        "county_POCAHONTAS",
        "county_POLK",
        "county_POTTAWATTAMIE",
        "county_POWESHIEK",
        "county_RINGGOLD",
        "county_SAC",
        "county_SCOTT",
        "county_SHELBY",
        "county_SIOUX",
        "county_STORY",
        "county_TAMA",
        "county_TAYLOR",
        "county_UNION",
        "county_VAN BUREN",
        "county_WAPELLO",
        "county_WARREN",
        "county_WASHINGTON",
        "county_WAYNE",
        "county_WEBSTER",
        "county_WINNEBAGO",
        "county_WINNESHIEK",
        "county_WOODBURY",
        "county_WORTH",
        "county_WRIGHT",
    ]

    payload_df_encoded = pd.get_dummies(payload_df, columns=["county"], prefix="county")

    payload_cols = set(payload_df_encoded.columns)
    missing_cols = set(feature_cols) - payload_cols
    for col in missing_cols:
        payload_df_encoded[col] = 0
    extra_cols = payload_cols - set(feature_cols)
    for col in extra_cols:
        payload_df_encoded.drop(col, axis=1, inplace=True)
    return payload_df_encoded[feature_cols]


def get_data_forecast_request(payload: ForecastRequest) -> pd.DataFrame:
    try:
        bq_client = bigquery.Client()

        # Fixed: Removed conflicting hardcoded strings, syntax string quotes, and parameterized SQL properly
        # Looks back 45 days to calculate a clean rolling window of 30 days
        query = """
            SELECT 
                invoice_and_item_number,
                date,
                county,
                bottles_sold,
                sale_dollars,
                volume_sold_liters
            FROM `bigquery-public-data.iowa_liquor_sales.sales` 
            WHERE date BETWEEN DATE_SUB(PARSE_DATE('%Y-%m-%d', @target_date), INTERVAL 32 DAY) AND DATE_ADD(PARSE_DATE('%Y-%m-%d', @target_date), INTERVAL 1 DAY)
              AND county = UPPER(@county)
            ORDER BY date ASC
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("county", "STRING", payload.county),
                bigquery.ScalarQueryParameter(
                    "target_date", "STRING", payload.target_date
                ),
            ]
        )

        logger.info(
            f"Executing BigQuery pull for county: {payload.county} around execution date: {payload.target_date}"
        )
        query_job = bq_client.query(query, job_config=job_config)
        historical_df = query_job.to_dataframe()

        if historical_df.empty:
            raise HTTPException(
                status_code=400,
                detail=f"No data returned from BigQuery for county {payload.county} in specified historical range.",
            )

        return prepare_time_series_data(historical_df)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error querying BigQuery: {str(e)}"
        )


def prepare_time_series_data(df):
    """Preparar datos para forecasting de series temporales"""
    logger.info("Preparando datos para series temporales...")

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

    # Fill NaNs from early lag steps gracefully with zero or mean to avoid dropping evaluation row
    df_daily = df_daily.fillna(0)

    return df_daily
