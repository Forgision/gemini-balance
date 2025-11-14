import pytest

def test_get_keys_paginated_success(mock_verify_auth_token, client, mock_key_manager):
    """Test successful retrieval of paginated keys with default parameters."""
    setattr(
        mock_key_manager.get_all_keys_with_fail_count,
        "return_value",
        {
            "valid_keys": {
                "test_key_1": {"status": "active", "exhausted": False},
                "test_key_2": {"status": "active", "exhausted": False},
            },
            "invalid_keys": {"test_key_3": {"status": "inactive", "exhausted": False}},
        },
    )

    response = client.get(
        "/api/keys",
        cookies={"auth_token": "test_auth_token"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total_items"] == 3
    assert "test_key_1" in data["keys"]
    assert "test_key_3" in data["keys"]
    assert data["keys"]["test_key_1"] == 0


@pytest.mark.no_mock_auth
def test_get_keys_paginated_unauthorized(client, mock_key_manager):
    """Test unauthorized access to paginated keys."""
    response = client.get("/api/keys")
    assert response.status_code == 401


def test_get_keys_paginated_filter_by_status(mock_verify_auth_token, client, mock_key_manager):
    """Test filtering keys by status (valid/invalid)."""
    mock_key_manager.get_all_keys_with_fail_count.return_value = {  # type: ignore
        "valid_keys": {"valid_key": {"status": "active", "exhausted": False}},
        "invalid_keys": {"invalid_key": {"status": "inactive", "exhausted": False}},
    }

    # Test 'valid' status
    response_valid = client.get(
        "/api/keys?status=valid",
        cookies={"auth_token": "test_auth_token"},
    )
    assert response_valid.status_code == 200
    data_valid = response_valid.json()
    assert data_valid["total_items"] == 1
    assert "valid_key" in data_valid["keys"]
    assert "invalid_key" not in data_valid["keys"]

    # Test 'invalid' status
    response_invalid = client.get(
        "/api/keys?status=invalid",
        cookies={"auth_token": "test_auth_token"},
    )
    assert response_invalid.status_code == 200
    data_invalid = response_invalid.json()
    assert data_invalid["total_items"] == 1
    assert "invalid_key" in data_invalid["keys"]
    assert "valid_key" not in data_invalid["keys"]


def test_get_keys_paginated_search(mock_verify_auth_token, client, mock_key_manager):
    """Test searching for a specific key."""
    mock_key_manager.get_all_keys_with_fail_count.return_value = {  # type: ignore
        "valid_keys": {
            "search_target_key": {"status": "active", "exhausted": False},
            "another_key": {"status": "active", "exhausted": False},
        },
        "invalid_keys": {},
    }

    response = client.get(
        "/api/keys?search=target",
        cookies={"auth_token": "test_auth_token"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total_items"] == 1
    assert "search_target_key" in data["keys"]
    assert "another_key" not in data["keys"]


def test_get_keys_paginated_fail_count_threshold(mock_verify_auth_token, client, mock_key_manager):
    """Test filtering by fail count threshold."""
    mock_key_manager.get_all_keys_with_fail_count.return_value = {  # type: ignore
        "valid_keys": {"key_low_fail": {"status": "active", "exhausted": False}},
        "invalid_keys": {"key_high_fail": {"status": "inactive", "exhausted": False}},
    }

    # In v2, status "active" maps to fail_count=0, "inactive" maps to fail_count=1
    # So threshold=1 should only return inactive keys
    response = client.get(
        "/api/keys?fail_count_threshold=1",
        cookies={"auth_token": "test_auth_token"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total_items"] == 1
    assert "key_high_fail" in data["keys"]
    assert "key_low_fail" not in data["keys"]


def test_get_keys_paginated_pagination(mock_verify_auth_token, client, mock_key_manager):
    """Test the pagination logic."""
    keys = {f"key_{i}": {"status": "active", "exhausted": False} for i in range(20)}
    mock_key_manager.get_all_keys_with_fail_count.return_value = {  # type: ignore
        "valid_keys": keys,
        "invalid_keys": {},
    }

    # Get page 2 with a limit of 5
    response = client.get(
        "/api/keys?page=2&limit=5",
        cookies={"auth_token": "test_auth_token"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total_items"] == 20
    assert data["total_pages"] == 4
    assert data["current_page"] == 2
    assert len(data["keys"]) == 5
    # The keys should be key_5 through key_9
    assert "key_5" in data["keys"]
    assert "key_9" in data["keys"]
    assert "key_4" not in data["keys"]
    assert "key_10" not in data["keys"]


# Tests for get_all_keys
def test_get_all_keys_success(mock_verify_auth_token, client, mock_key_manager):
    """Test successful retrieval of all keys for bulk operations."""
    mock_key_manager.get_all_keys_with_fail_count.return_value = {  # type: ignore
        "valid_keys": {
            "valid_1": {"status": "active", "exhausted": False},
            "valid_2": {"status": "active", "exhausted": False},
        },
        "invalid_keys": {"invalid_1": {"status": "inactive", "exhausted": False}},
    }

    response = client.get(
        "/api/keys/all",
        cookies={"auth_token": "test_auth_token"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] == 3
    assert "valid_1" in data["valid_keys"]
    assert "invalid_1" in data["invalid_keys"]
    assert len(data["valid_keys"]) == 2
    assert len(data["invalid_keys"]) == 1


@pytest.mark.no_mock_auth
def test_get_all_keys_unauthorized(client, mock_key_manager):
    """Test unauthorized access to the get_all_keys endpoint."""
    response = client.get("/api/keys/all")
    assert response.status_code == 401
