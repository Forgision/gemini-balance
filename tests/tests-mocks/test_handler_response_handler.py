from app.handler.response_handler import (
    GeminiResponseHandler,
    OpenAIResponseHandler,
    _handle_gemini_stream_response,
    _handle_gemini_normal_response,
    _handle_openai_stream_response,
    _handle_openai_normal_response,
    _extract_result,
)

# Tests for GeminiResponseHandler
def test_gemini_response_handler_stream():
    """Test GeminiResponseHandler with a streaming response."""
    handler = GeminiResponseHandler()
    response = {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}
    result = handler.handle_response(response, "gemini-pro", stream=True)
    assert "candidates" in result

def test_gemini_response_handler_normal():
    """Test GeminiResponseHandler with a normal response."""
    handler = GeminiResponseHandler()
    response = {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}
    result = handler.handle_response(response, "gemini-pro", stream=False)
    assert "candidates" in result

# Tests for OpenAIResponseHandler
def test_openai_response_handler_stream():
    """Test OpenAIResponseHandler with a streaming response."""
    handler = OpenAIResponseHandler(config=None)
    response = {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}
    result = handler.handle_response(response, "gemini-pro", stream=True)
    assert "choices" in result

def test_openai_response_handler_normal():
    """Test OpenAIResponseHandler with a normal response."""
    handler = OpenAIResponseHandler(config=None)
    response = {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}
    result = handler.handle_response(response, "gemini-pro", stream=False)
    assert "choices" in result

# Tests for private helper functions
def test_handle_gemini_stream_response():
    """Test _handle_gemini_stream_response."""
    response = {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}
    result = _handle_gemini_stream_response(response, "gemini-pro", stream=True)
    assert "candidates" in result

def test_handle_gemini_normal_response():
    """Test _handle_gemini_normal_response."""
    response = {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}
    result = _handle_gemini_normal_response(response, "gemini-pro", stream=False)
    assert "candidates" in result

def test_handle_openai_stream_response():
    """Test _handle_openai_stream_response."""
    response = {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}
    result = _handle_openai_stream_response(response, "gemini-pro", "stop", None)
    assert "choices" in result

def test_handle_openai_normal_response():
    """Test _handle_openai_normal_response."""
    response = {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}
    result = _handle_openai_normal_response(response, "gemini-pro", "stop", None)
    assert "choices" in result

def test_extract_result_stream():
    """Test _extract_result with a streaming response."""
    response = {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}
    text, _, _, _ = _extract_result(response, "gemini-pro", stream=True)
    assert text == "Hello"

def test_extract_result_normal():
    """Test _extract_result with a normal response."""
    response = {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}
    text, _, _, _ = _extract_result(response, "gemini-pro", stream=False)
    assert text == "Hello"
