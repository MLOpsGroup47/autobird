import os
import random

from locust import HttpUser, between, task

from tests.performancetests.create_dummy_wav import create_dummy_wav

# Pre-load data to separate IO from load testing logic
AUDIO_FILENAME = "dummy_audio.wav"
AUDIO_PATH = os.path.join(os.path.dirname(__file__), AUDIO_FILENAME)

# Generate if not exists
if not os.path.exists(AUDIO_PATH):
    create_dummy_wav(AUDIO_PATH)

# Read into memory
with open(AUDIO_PATH, "rb") as f:
    AUDIO_DATA = f.read()


class MyUser(HttpUser):
    """A simple Locust user class that defines the tasks to be performed by the users."""

    wait_time = between(1, 2)

    @task
    def get_root(self) -> None:
        """A task that simulates a user visiting the root URL of the FastAPI app."""
        self.client.get("/")

    
    @task(2)
    def get_files(self) -> None:
        """A task that simulates a user visiting a random item URL of the FastAPI app."""
        self.client.get(f"/files/")

    @task(1)
    def get_classify(self) -> None:
        """A task that simulates a user classifying an audio file."""
        self.client.post(
            "/classify/", 
            files={"audio": (AUDIO_FILENAME, AUDIO_DATA, "audio/wav")}
        )