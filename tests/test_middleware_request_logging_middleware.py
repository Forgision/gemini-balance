import json
from unittest.mock import patch

import pytest
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse, Response
from starlette.testclient import TestClient

from app.middleware.request_logging_middleware import RequestLoggingMiddleware


@pytest.fixture
def mock_logger():
    """Fixture to patch the logger used in the middleware."""
    with patch("app.middleware.request_logging_middleware.logger") as mock:
        yield mock


@pytest.fixture
def test_app():
    """Fixture to create a FastAPI app with the logging middleware."""
    app = FastAPI()

    @app.post("/test")
    async def test_endpoint(request: Request):
        body = await request.body()
        return Response(content=body)

    @app.post("/no_body")
    async def no_body_endpoint():
        return JSONResponse({"status": "ok"})

    app.add_middleware(RequestLoggingMiddleware)
    return app


@pytest.fixture
def client(test_app):
    """Fixture to create a TestClient for the app."""
    return TestClient(test_app)


def test_logging_middleware_logs_request(client, mock_logger):
    # Given
    test_body = {"key": "value"}
    test_body_bytes = json.dumps(test_body).encode()

    # When
    response = client.post("/test", content=test_body_bytes)

    # Then
    assert response.status_code == 200
    assert response.content == test_body_bytes
    mock_logger.info.assert_any_call("Request path: /test")
    mock_logger.info.assert_any_call(
        f"Formatted request body:\n{json.dumps(test_body, indent=2, ensure_ascii=False)}"
    )


def test_logging_middleware_invalid_json(client, mock_logger):
    # Given
    invalid_json_body = b'{"key": "value"'

    # When
    response = client.post("/test", content=invalid_json_body)

    # Then
    assert response.status_code == 200
    mock_logger.info.assert_any_call("Request path: /test")
    mock_logger.error.assert_called_with("Request body is not valid JSON.")


def test_logging_middleware_no_body(client, mock_logger):
    # When
    response = client.post("/no_body")

    # Then
    assert response.status_code == 200
    mock_logger.info.assert_called_with("Request path: /no_body")
    # Assert that no error or formatted body logging occurred
    mock_logger.error.assert_not_called()
    for call in mock_logger.info.call_args_list:
        assert "Formatted request body" not in call[0][0]


@pytest.mark.asyncio
async def test_logging_middleware_body_read_error(mock_logger):
    # Given
    app = FastAPI()  # A dummy app for middleware instantiation
    middleware = RequestLoggingMiddleware(app)

    async def receive():
        raise ValueError("Test error")

    mock_request = Request(
        {"type": "http", "method": "POST", "path": "/test_read_error", "headers": []},
        receive,
    )

    async def call_next(request):
        # This won't be called if the body read fails, but it's required by the middleware dispatch signature.
        return JSONResponse({"status": "ok"})

    # When
    await middleware.dispatch(mock_request, call_next)

    # Then
    mock_logger.error.assert_called_once_with("Error reading request body: Test error")


def test_request_body_can_be_read_after_middleware(client):
    # Given
    app = client.app

    @app.post("/read_body")
    async def read_body_endpoint(request: Request):
        body = await request.json()
        return JSONResponse({"received_body": body})

    test_body = {"message": "hello"}
    test_body_bytes = json.dumps(test_body).encode()

    # When
    response = client.post("/read_body", content=test_body_bytes)

    # Then
    assert response.status_code == 200
    assert response.json() == {"received_body": test_body}
