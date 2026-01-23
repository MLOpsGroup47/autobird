from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.app.inference import app


@pytest.fixture
def mock_dependencies():
    """Mock external dependencies globally for these tests."""
    with patch("api.app.inference.torch.load") as mock_load, \
         patch("api.app.inference.Model") as mock_model_cls, \
         patch("api.app.inference.predict_file") as mock_predict, \
         patch("api.app.inference.inference_load") as mock_inf_load, \
         patch("api.app.inference.sf.read") as mock_sf_read:
         
        # Setup mock return values for model loading
        mock_load.return_value = {
            "n_classes": 10,
            "hp": {"d_model": 64, "n_heads": 2, "n_layers": 1},
            "model_state": {}
        }
        
        # Mock Model instance
        mock_model_instance = MagicMock()
        mock_model_cls.return_value = mock_model_instance
        
        # Mock prediction result
        mock_predict.return_value = {"label": "Andean Guan"}

        # Mock soundfile read output for the endpoint (data, samplerate)
        mock_sf_read.return_value = (np.zeros((1000,)), 16000)
        
        yield

def test_read_root(mock_dependencies):
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"Hello": "Welcome to AutoBird Inference API",
            "Help":"Use /classify/ endpoint to POST an audio file for classification."}

def test_classify(mock_dependencies):
    with TestClient(app) as client:
        # We simulate the file upload. 
        # The content doesn't matter much since sf.read is mocked within the app,
        # but we need to provide a file-like object for FastAPI to accept the request.
        files = {'audio': ('test_audio.mp3', b'fake_mp3_bytes', 'audio/mpeg')}
        
        response = client.post("/classify/", files=files)
             
        assert response.status_code == 200
        
        assert isinstance(response.json(), str)
