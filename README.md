# Taligent Serving

A machine learning forecasting service for Iowa liquor sales predictions using time series analysis and XGBoost/Prophet models.

## Overview

This project provides a FastAPI-based REST API for forecasting liquor sales across Iowa counties. It leverages historical sales data from BigQuery, applies advanced time series feature engineering, and serves predictions using trained XGBoost models.

## Features

- **Time Series Forecasting**: Predict daily liquor sales for Iowa counties
- **Multiple Forecast Types**: Single-day predictions and weekly forecasts
- **Advanced Feature Engineering**: Lag features, rolling windows, and temporal features
- **BigQuery Integration**: Real-time data fetching from Google BigQuery public datasets
- **Cloud Native**: Designed for Google Cloud Run with GCS model storage
- **Model Training Pipeline**: Complete training workflow with XGBoost and Prophet
- **Comprehensive Testing**: Unit tests for API endpoints, data processing, and model loading

## Architecture

```
taligent_serving/
├── serving/
│   ├── main.py              # FastAPI application & prediction endpoints
│   ├── dockerfile           # Container configuration
│   └── requirements.txt     # Python dependencies
├── training/
│   └── training.py          # Model training pipeline
├── tests/
│   ├── test_main.py         # API endpoint tests
│   ├── test_utils.py        # Data processing tests
│   └── test_model_loading.py # Model loading tests
├── models/                  # Trained model storage
├── results/                 # Training reports
└── pyproject.toml          # Project configuration
```

## Tech Stack

- **Language**: Python 3.13+
- **API Framework**: FastAPI with Uvicorn
- **ML Models**: XGBoost, Prophet
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Cloud Services**: Google BigQuery, Google Cloud Storage
- **Package Management**: Poetry
- **Testing**: Pytest, pytest-cov, httpx

## Installation

### Prerequisites

- Python 3.13 or higher
- Poetry (for dependency management)
- Google Cloud credentials (for BigQuery and GCS access)

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd taligent_serving
```

2. **Install dependencies**
```bash
poetry install
```

3. **Configure environment variables**
```bash
# Create .env file
GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json
GCS_MODEL_PATH=gs://your-bucket/models/best_forecast_model.pkl
FORECAST_MODEL=xgboost  # or 'prophet'
FORECAST_HORIZON=30
```

## Usage

### Running the API Server

**Development:**
```bash
poetry run uvicorn serving.main:app --reload --host 0.0.0.0 --port 8000
```

**Production (Docker):**
```bash
docker build -t taligent-serving .
docker run -p 8000:8000 taligent-serving
```

### API Endpoints

#### 1. Health Check
```bash
GET /healthz
```

Response:
```json
{
  "status": "healthy"
}
```

#### 2. Single Day Prediction
```bash
POST /predict
Content-Type: application/json

{
  "target_date": "2025-02-15",
  "county": "POLK"
}
```

Response:
```json
{
  "target_date": "2025-02-15",
  "county": "POLK",
  "forecast": [500.0]
}
```

#### 3. Weekly Forecast
```bash
GET /weekly
```

Response:
```json
{
  "county": "POLK",
  "daily_breakdown": [450.0, 475.0, 500.0, ...],
  "total_weekly_forecast": 3500.0
}
```

## Model Training

### Training a New Model

```bash
poetry run python training/training.py
```

### Training Configuration

Configure training behavior via environment variables:

- `FORECAST_MODEL`: Model type to train (`xgboost` or `prophet`)
- `FORECAST_HORIZON`: Number of days to forecast (default: 30)
- `BEST_MODEL_METRIC`: Metric for model selection (`validation_rmse`)
- `BEST_MODEL_MODE`: Optimization mode (`min` for RMSE, `max` for R²)

### Training Pipeline

1. **Data Loading**: Fetches historical sales data from BigQuery
2. **Feature Engineering**: Creates temporal features, lag features, and rolling windows
3. **Model Training**: Trains XGBoost or Prophet models
4. **Evaluation**: Calculates RMSE, MAE, R², MAPE, and SMAPE metrics
5. **Model Saving**: Stores trained model and metadata to `models/` directory

### Feature Engineering

The pipeline creates the following features:

**Temporal Features:**
- Year, month, day, day of week, day of year, week of year
- Weekend indicator, month start/end, quarter start/end

**Lag Features:**
- Sales lagged by 1, 7, 14, and 30 days

**Rolling Window Features:**
- 7, 14, and 30 day rolling mean and standard deviation

**Categorical Features:**
- One-hot encoded Iowa counties (99 counties)

## Testing

### Run All Tests
```bash
poetry run pytest
```

### Run with Coverage
```bash
poetry run pytest --cov=serving --cov-report=html
```

### Run Specific Test File
```bash
poetry run pytest tests/test_main.py
```

### Test Structure

- `test_main.py`: FastAPI endpoint tests
- `test_utils.py`: Data processing and feature engineering tests
- `test_model_loading.py`: Model loading and GCS download tests

## Deployment

### Google Cloud Run

1. **Build and push container**
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/taligent-serving
```

2. **Deploy to Cloud Run**
```bash
gcloud run deploy taligent-serving \
  --image gcr.io/PROJECT_ID/taligent-serving \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Environment Variables for Production

Set the following environment variables in Cloud Run:

- `GCS_MODEL_PATH`: GCS path to trained model
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to service account credentials
- `FORECAST_MODEL`: Model type to use
- `FORECAST_HORIZON`: Forecast horizon in days

## Data Source

The service uses the **Iowa Liquor Sales** dataset from Google BigQuery public datasets:

- **Dataset**: `bigquery-public-data.iowa_liquor_sales.sales`
- **Coverage**: Statewide liquor sales from 2012 to present
- **Granularity**: Individual transaction-level data
- **Geography**: 99 Iowa counties

## Model Performance

Current model metrics (from training reports):

- **Validation R²**: ~0.85
- **Validation RMSE**: ~$200-300
- **Validation MAE**: ~$150-250
- **Validation MAPE**: ~15-20%

*Note: Actual metrics vary based on training data and model configuration.*

## Development

### Code Style

The project uses:
- **Ruff** for linting and formatting
- **MyPy** for type checking
- **Pre-commit hooks** for code quality

### Pre-commit Setup

```bash
poetry run pre-commit install
```

### Adding New Features

1. Add feature engineering logic in `serving/main.py` or `training/training.py`
2. Update tests in `tests/` directory
3. Run tests to ensure compatibility
4. Update documentation

## Troubleshooting

### Common Issues

**Model Loading Fails:**
- Ensure `GCS_MODEL_PATH` is set correctly
- Verify Google Cloud credentials are configured
- Check model file exists in specified location

**BigQuery Connection Errors:**
- Verify `GOOGLE_APPLICATION_CREDENTIALS` is set
- Ensure service account has BigQuery access
- Check network connectivity

**Prediction Errors:**
- Verify target date format (YYYY-MM-DD)
- Ensure county name matches BigQuery data (uppercase)
- Check that historical data exists for the requested date range

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run tests and ensure they pass
6. Submit a pull request

## License

[Add your license information here]

## Authors

- Diego Gasch - [eldie1984@gmail.com](mailto:eldie1984@gmail.com)

## Acknowledgments

- Iowa Department of Revenue for the liquor sales data
- Google Cloud Platform for BigQuery public datasets
- XGBoost and Prophet communities for excellent ML libraries
