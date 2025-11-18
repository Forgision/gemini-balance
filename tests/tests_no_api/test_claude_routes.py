"""
Integration tests for Claude proxy routes.
"""

import pytest


@pytest.mark.asyncio
async def test_claude_create_message(test_client, auth_header):
    """Test creating message via Claude proxy."""
    payload = {
        "model": "gemini-pro",
        "messages": [
            {
                "role": "user",
                "content": "Hello"
            }
        ],
        "max_tokens": 100,
        "stream": False
    }
    
    response = test_client.post("/claude/v1/messages", json=payload, headers=auth_header)
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)


@pytest.mark.asyncio
async def test_claude_count_tokens(test_client, auth_header):
    """Test counting tokens via Claude proxy."""
    payload = {
        "model": "gemini-pro",
        "messages": [
            {
                "role": "user",
                "content": "Count these tokens"
            }
        ]
    }
    
    response = test_client.post("/claude/v1/messages/count_tokens", json=payload, headers=auth_header)
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)

