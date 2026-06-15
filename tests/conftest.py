import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_historical_data():
    """Create sample historical data for testing"""
    dates = pd.date_range(start="2025-01-01", periods=45, freq="D")
    data = {
        "invoice_and_item_number": [f"INV{i}" for i in range(45)],
        "date": dates,
        "county": ["POLK"] * 45,
        "bottles_sold": np.random.randint(10, 100, 45),
        "sale_dollars": np.random.randint(100, 1000, 45),
        "volume_sold_liters": np.random.randint(5, 50, 45),
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_time_series_data():
    """Create sample time series data after preparation"""
    dates = pd.date_range(start="2025-01-01", periods=45, freq="D")
    data = {
        "ds": dates,
        "county": ["POLK"] * 45,
        "y": np.random.randint(100, 1000, 45),
        "bottles": np.random.randint(10, 100, 45),
        "liters": np.random.randint(5, 50, 45),
        "transactions": np.random.randint(1, 20, 45),
    }
    df = pd.DataFrame(data)

    # Add time features
    df["year"] = df["ds"].dt.year
    df["month"] = df["ds"].dt.month
    df["day"] = df["ds"].dt.day
    df["dayofweek"] = df["ds"].dt.dayofweek
    df["dayofyear"] = df["ds"].dt.dayofyear
    df["weekofyear"] = df["ds"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_month_start"] = df["ds"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["ds"].dt.is_month_end.astype(int)
    df["is_quarter_start"] = df["ds"].dt.is_quarter_start.astype(int)
    df["is_quarter_end"] = df["ds"].dt.is_quarter_end.astype(int)

    # Add lag features
    for lag in [1, 7, 14, 30]:
        df[f"y_lag_{lag}"] = df["y"].shift(lag).fillna(0)

    # Add rolling features
    for window in [7, 14, 30]:
        df[f"y_rolling_mean_{window}"] = (
            df["y"].rolling(window=window, min_periods=1).mean()
        )
        df[f"y_rolling_std_{window}"] = (
            df["y"].rolling(window=window, min_periods=1).std().fillna(0)
        )

    return df.fillna(0)


@pytest.fixture
def mock_model():
    """Create a mock model for testing"""
    model = Mock()
    model.predict = Mock(return_value=np.array([500.0]))
    return model


@pytest.fixture
def mock_bigquery_client():
    """Create a mock BigQuery client"""
    client = Mock()
    query_job = Mock()
    query_job.to_dataframe = Mock(return_value=pd.DataFrame())
    client.query = Mock(return_value=query_job)
    return client


@pytest.fixture
def mock_storage_client():
    """Create a mock GCS storage client"""
    client = Mock()
    bucket = Mock()
    blob = Mock()
    bucket.blob = Mock(return_value=blob)
    client.bucket = Mock(return_value=bucket)
    return client


@pytest.fixture
def sample_forecast_request():
    """Create a sample forecast request"""
    from serving.main import ForecastRequest

    return ForecastRequest(target_date="2025-02-15", county="POLK")


@pytest.fixture
def mock_model_obj():
    """Create a mock model object for testing"""
    model = Mock()
    model.predict = Mock(return_value=np.array([500.0]))
    return model
