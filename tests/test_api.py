from fastapi.testclient import TestClient

from api.app.inference import app

client = TestClient(app)


def test_read_root():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"Hello": "Welcome to AutoBird Inference API",
            "Help":"Use /classify/ endpoint to POST an audio file for classification."}

def test_classify():
    with TestClient(app) as client:
        # Use a sample audio file for testing
        audio_file_path = "data/voice_of_birds/Andean Guan_sound/Andean Guan2.mp3"
        with open(audio_file_path, "rb") as audio_file:
            response = client.post(
                "/classify/",
                files={"audio": ("Andean Guan2.mp3", audio_file, "audio/mpeg")},
            )
        assert response.status_code == 200
        json_response = response.json()
        assert isinstance(json_response, str)