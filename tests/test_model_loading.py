import pytest
from unittest.mock import Mock, patch
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serving.main import download_model, load_model


class TestDownloadModel:
    """Test download_model function"""

    @patch("serving.main.storage.Client")
    def test_download_model_success(self, mock_storage_client):
        """Test successful model download from GCS"""
        gcs_path = "gs://test-bucket/models/model.pkl"
        local_path = "/tmp/test_model.pkl"

        # Setup mock
        mock_bucket = Mock()
        mock_blob = Mock()
        mock_bucket.blob = Mock(return_value=mock_blob)
        mock_storage_client.return_value.bucket = Mock(return_value=mock_bucket)

        result = download_model(gcs_path, local_path)

        # Verify the blob download was called
        mock_blob.download_to_filename.assert_called_once_with(local_path)
        assert result == local_path

    @patch("serving.main.storage.Client")
    def test_download_model_path_parsing(self, mock_storage_client):
        """Test that GCS path is correctly parsed into bucket and blob path"""
        gcs_path = "gs://my-bucket/path/to/model.pkl"
        local_path = "/tmp/model.pkl"

        mock_bucket = Mock()
        mock_blob = Mock()
        mock_bucket.blob = Mock(return_value=mock_blob)
        mock_storage_client.return_value.bucket = Mock(return_value=mock_bucket)

        download_model(gcs_path, local_path)

        # Verify bucket name extraction
        mock_storage_client.return_value.bucket.assert_called_once_with("my-bucket")
        # Verify blob path extraction
        mock_bucket.blob.assert_called_once_with("path/to/model.pkl")

    @patch("serving.main.storage.Client")
    def test_download_model_gs_prefix_removal(self, mock_storage_client):
        """Test that gs:// prefix is removed from path"""
        gcs_path = "gs://test-bucket/model.pkl"

        mock_bucket = Mock()
        mock_blob = Mock()
        mock_bucket.blob = Mock(return_value=mock_blob)
        mock_storage_client.return_value.bucket = Mock(return_value=mock_bucket)

        download_model(gcs_path)

        # Verify that the path was parsed correctly (gs:// removed)
        mock_storage_client.return_value.bucket.assert_called_once_with("test-bucket")


class TestLoadModel:
    """Test load_model function"""

    @patch("serving.main.joblib.load")
    @patch("serving.main.download_model")
    @patch.dict(os.environ, {"GCS_MODEL_PATH": "gs://test-bucket/model.pkl"})
    def test_load_model_from_gcs(self, mock_download, mock_joblib_load):
        """Test loading model from GCS when GCS_MODEL_PATH is set"""
        mock_model = Mock()
        mock_joblib_load.return_value = mock_model
        mock_download.return_value = "/tmp/model.pkl"

        # Reset global variables
        import serving.main

        serving.main.MODEL = None
        serving.main.METADATA = None

        load_model()

        # Verify download was called
        mock_download.assert_called_once_with("gs://test-bucket/model.pkl")
        # Verify joblib.load was called
        mock_joblib_load.assert_called_once()
        # Verify global variables were set
        assert serving.main.MODEL == mock_model

    @patch("serving.main.joblib.load")
    @patch.dict(os.environ, {}, clear=True)
    def test_load_model_local_fallback(self, mock_joblib_load):
        """Test loading model from local path when GCS_MODEL_PATH is not set"""
        mock_model = Mock()
        mock_joblib_load.return_value = mock_model

        # Reset global variables
        import serving.main

        serving.main.MODEL = None
        serving.main.METADATA = None

        load_model()

        # Verify joblib.load was called with local path
        mock_joblib_load.assert_called_once_with("./models/best_forecast_model.pkl")
        # Verify global variables were set
        assert serving.main.MODEL == mock_model

    @patch("serving.main.joblib.load")
    def test_load_model_sets_globals(self, mock_joblib_load):
        """Test that load_model sets global MODEL and METADATA variables"""
        mock_model = Mock()
        mock_joblib_load.return_value = mock_model

        # Reset global variables
        import serving.main

        serving.main.MODEL = None
        serving.main.METADATA = None

        load_model()

        # Verify globals are set
        assert serving.main.MODEL is not None
        assert serving.main.METADATA is None

    @patch("serving.main.joblib.load")
    def test_load_model_file_not_found(self, mock_joblib_load):
        """Test handling when model file is not found"""
        mock_joblib_load.side_effect = FileNotFoundError("Model file not found")

        # Reset global variables
        import serving.main

        serving.main.MODEL = None
        serving.main.METADATA = None

        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            load_model()

    @patch("serving.main.joblib.load")
    def test_load_model_corrupted_file(self, mock_joblib_load):
        """Test handling when model file is corrupted"""
        mock_joblib_load.side_effect = Exception("Corrupted file")

        # Reset global variables
        import serving.main

        serving.main.MODEL = None
        serving.main.METADATA = None

        # Should raise the exception
        with pytest.raises(Exception):
            load_model()
