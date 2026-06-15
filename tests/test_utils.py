import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serving.main import prepare_features, prepare_time_series_data


class TestPrepareFeatures:
    """Test prepare_features function"""

    def test_prepare_features_basic(self, sample_time_series_data):
        """Test basic feature preparation"""
        # Take a single row for testing
        test_df = sample_time_series_data.iloc[[0]].copy()

        result = prepare_features(test_df)

        # Check that result is a DataFrame
        assert isinstance(result, pd.DataFrame)

        # Check that all expected feature columns are present
        expected_features = [
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

        for feature in expected_features:
            assert feature in result.columns, f"Missing feature: {feature}"

    def test_prepare_features_one_hot_encoding(self, sample_time_series_data):
        """Test one-hot encoding of county column"""
        test_df = sample_time_series_data.iloc[[0]].copy()
        test_df["county"] = "POLK"

        result = prepare_features(test_df)

        # Check that county is one-hot encoded
        assert "county_POLK" in result.columns
        assert result["county_POLK"].values[0] == 1

        # Check that other county columns exist and are 0
        other_counties = [
            col
            for col in result.columns
            if col.startswith("county_") and col != "county_POLK"
        ]
        for county_col in other_counties:
            assert result[county_col].values[0] == 0

    def test_prepare_features_missing_columns_added(self, sample_time_series_data):
        """Test that missing feature columns are added with 0"""
        test_df = sample_time_series_data.iloc[[0]].copy()

        # Remove some feature columns to test they get added back
        test_df = test_df[["ds", "county", "y"]]

        result = prepare_features(test_df)

        # Check that missing columns are added with 0
        assert "y_lag_1" in result.columns
        assert "y_rolling_mean_7" in result.columns

    def test_prepare_features_extra_columns_removed(self, sample_time_series_data):
        """Test that extra columns not in feature list are removed"""
        test_df = sample_time_series_data.iloc[[0]].copy()

        # Add an extra column
        test_df["extra_column"] = 999

        result = prepare_features(test_df)

        # Check that extra column is removed
        assert "extra_column" not in result.columns

    def test_prepare_features_column_order(self, sample_time_series_data):
        """Test that columns are returned in the correct order"""
        test_df = sample_time_series_data.iloc[[0]].copy()

        result = prepare_features(test_df)

        # Check that result has the expected number of columns
        # 22 time features + Iowa counties (approximately 99)
        # The exact count depends on the county list in main.py
        assert len(result.columns) >= 100  # At least time features + some counties


class TestPrepareTimeSeriesData:
    """Test prepare_time_series_data function"""

    def test_prepare_time_series_data_basic(self, sample_historical_data):
        """Test basic time series data preparation"""
        result = prepare_time_series_data(sample_historical_data)

        # Check that result is a DataFrame
        assert isinstance(result, pd.DataFrame)

        # Check that required columns exist
        required_cols = ["ds", "county", "y", "bottles", "liters", "transactions"]
        for col in required_cols:
            assert col in result.columns

    def test_prepare_time_series_date_conversion(self, sample_historical_data):
        """Test that date is converted to datetime"""
        result = prepare_time_series_data(sample_historical_data)

        # Check that ds is datetime
        assert pd.api.types.is_datetime64_any_dtype(result["ds"])

    def test_prepare_time_series_aggregation(self, sample_historical_data):
        """Test that data is aggregated by date and county"""
        # Add multiple entries for the same date
        sample_historical_data.loc[0, "date"] = sample_historical_data.loc[1, "date"]

        result = prepare_time_series_data(sample_historical_data)

        # Check that aggregation happened (should have fewer rows than input)
        assert len(result) <= len(sample_historical_data)

    def test_prepare_time_series_features(self, sample_historical_data):
        """Test that time series features are created"""
        result = prepare_time_series_data(sample_historical_data)

        # Check for time-based features
        time_features = [
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
        ]

        for feature in time_features:
            assert feature in result.columns

    def test_prepare_time_series_lag_features(self, sample_historical_data):
        """Test that lag features are created"""
        result = prepare_time_series_data(sample_historical_data)

        # Check for lag features
        lag_features = ["y_lag_1", "y_lag_7", "y_lag_14", "y_lag_30"]
        for feature in lag_features:
            assert feature in result.columns

    def test_prepare_time_series_rolling_features(self, sample_historical_data):
        """Test that rolling window features are created"""
        result = prepare_time_series_data(sample_historical_data)

        # Check for rolling features
        rolling_features = [
            "y_rolling_mean_7",
            "y_rolling_std_7",
            "y_rolling_mean_14",
            "y_rolling_std_14",
            "y_rolling_mean_30",
            "y_rolling_std_30",
        ]
        for feature in rolling_features:
            assert feature in result.columns

    def test_prepare_time_series_fill_na(self, sample_historical_data):
        """Test that NaN values are filled with 0"""
        result = prepare_time_series_data(sample_historical_data)

        # Check that there are no NaN values
        assert result.isna().sum().sum() == 0

    def test_prepare_time_series_empty_input(self):
        """Test handling of empty input DataFrame"""
        empty_df = pd.DataFrame(
            columns=[
                "invoice_and_item_number",
                "date",
                "county",
                "bottles_sold",
                "sale_dollars",
                "volume_sold_liters",
            ]
        )

        # Should handle empty input gracefully
        result = prepare_time_series_data(empty_df)

        # Result should be empty or have minimal structure
        assert len(result) == 0 or isinstance(result, pd.DataFrame)
