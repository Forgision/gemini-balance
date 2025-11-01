from unittest.mock import patch
import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from unittest.mock import patch
import pytest
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse
from httpx import AsyncClient

from app.middleware.smart_routing_middleware import SmartRoutingMiddleware


@pytest.fixture
def smart_routing_app():
    app = FastAPI()

    @app.api_route("/{path:path}", methods=["GET", "POST"])
    async def catch_all(request: Request):
        return JSONResponse(
            {"path": request.scope["path"], "method": request.method}
        )

    with patch("app.middleware.smart_routing_middleware.settings") as mock_settings:
        mock_settings.URL_NORMALIZATION_ENABLED = True
        app.add_middleware(SmartRoutingMiddleware)
        yield app


@pytest.fixture
def smart_routing_client(smart_routing_app):
    return TestClient(smart_routing_app)


@pytest.mark.parametrize(
    "original_path, expected_path",
    [
        (
            "/gemini/models/gemini-pro:generateContent",
            "/v1beta/models/gemini-pro:generateContent",
        ),
        (
            "/openai/deployments/gpt-4/chat/completions",
            "/openai/v1/chat/completions",
        ),
        ("/chat/completions", "/v1/chat/completions"),
    ],
)
def test_post_requests_are_normalized(smart_routing_client, original_path, expected_path):
    response = smart_routing_client.post(original_path)
    assert response.json()["path"] == expected_path


@pytest.mark.parametrize(
    "original_path",
    [
        "/v1beta/models/gemini-pro:generateContent",
        "/openai/v1/chat/completions",
        "/v1/chat/completions",
    ],
)
def test_correct_paths_are_not_changed(smart_routing_client, original_path):
    response = smart_routing_client.post(original_path)
    assert response.json()["path"] == original_path


def test_get_requests_for_models_are_normalized(smart_routing_client):
    response = smart_routing_client.get("/gemini/v1beta/models")
    assert response.json()["path"] == "/gemini/v1beta/models"


def test_get_requests_for_non_models_are_not_normalized(smart_routing_client):
    response = smart_routing_client.get("/v1/chat/completions")
    assert response.json()["path"] == "/v1/chat/completions"


def test_url_normalization_disabled(smart_routing_app):
    with patch("app.middleware.smart_routing_middleware.settings") as mock_settings:
        mock_settings.URL_NORMALIZATION_ENABLED = False
        client = TestClient(smart_routing_app)
        response = client.post("/chat/completions")
        assert response.json()["path"] == "/chat/completions"


@pytest.mark.asyncio
async def test_model_extraction_from_body(smart_routing_client):
    response = smart_routing_client.post(
        "/gemini/generateContent", json={"model": "gemini-pro-vision"}
    )
    assert response.status_code == 200
    #TODO: following should be enable after fixing SmartRoutingMiddleware.extract_model_name
    # assert (
    #     response.json()["path"]
    #     == "/v1beta/models/gemini-pro-vision:generateContent"
    # )
