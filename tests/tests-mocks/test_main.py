from fastapi.testclient import TestClient
from app.main import app

def test_create_app():
    """Test that the app is created successfully."""
    assert app is not None

def test_read_main():
    """Test the main endpoint."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
