import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app import app


def test_home():
    app.config["TESTING"] = True
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200


def test_info():
    app.config["TESTING"] = True
    client = app.test_client()
    response = client.get("/info")
    assert response.status_code == 200