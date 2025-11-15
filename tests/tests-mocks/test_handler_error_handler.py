import pytest
from unittest.mock import MagicMock
from fastapi import HTTPException
from app.handler.error_handler import handle_route_errors

@pytest.mark.asyncio
async def test_handle_route_errors_success():
    """Test that the context manager executes successfully."""
    logger = MagicMock()
    operation_name = "test_operation"

    async with handle_route_errors(logger, operation_name):
        pass

    logger.info.assert_called_with("test_operation request successful")

@pytest.mark.asyncio
async def test_handle_route_errors_http_exception():
    """Test that HTTPException is re-raised."""
    logger = MagicMock()
    operation_name = "test_operation"

    with pytest.raises(HTTPException) as exc_info:
        async with handle_route_errors(logger, operation_name):
            raise HTTPException(status_code=400, detail="Test error")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Test error"
    logger.error.assert_called_once()

@pytest.mark.asyncio
async def test_handle_route_errors_other_exception():
    """Test that other exceptions are caught and raised as HTTPException."""
    logger = MagicMock()
    operation_name = "test_operation"

    with pytest.raises(HTTPException) as exc_info:
        async with handle_route_errors(logger, operation_name):
            raise ValueError("Test error")

    assert exc_info.value.status_code == 500
    assert "Internal server error" in exc_info.value.detail
    logger.error.assert_called_once()
