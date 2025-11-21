import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch
from app.service.key import key_manager_v1 as key_manager_module
from app.service.key.key_manager_v1 import get_key_manager_instance, reset_key_manager_instance
from datetime import datetime, timedelta

@pytest_asyncio.fixture(autouse=True)
async def fully_reset_key_manager(monkeypatch):
    """Fully resets the KeyManager singleton, clearing preserved state."""
    await reset_key_manager_instance()
    monkeypatch.setattr(key_manager_module, "_preserved_failure_counts", None)
    monkeypatch.setattr(key_manager_module, "_preserved_vertex_failure_counts", None)
    monkeypatch.setattr(key_manager_module, "_preserved_old_api_keys_for_reset", None)
    monkeypatch.setattr(key_manager_module, "_preserved_vertex_old_api_keys_for_reset", None)
    monkeypatch.setattr(key_manager_module, "_preserved_next_key_in_cycle", None)
    monkeypatch.setattr(key_manager_module, "_preserved_vertex_next_key_in_cycle", None)
    monkeypatch.setattr(key_manager_module, "_singleton_instance", None)
    
    # Mock database functions to avoid actual database calls
    mock_set_exhausted = AsyncMock(return_value=None)
    monkeypatch.setattr("app.service.key.key_manager_v1.set_key_exhausted_status", mock_set_exhausted)

@pytest_asyncio.fixture(scope="function")
async def key_manager():
    """Fixture to provide a KeyManager instance."""
    return await get_key_manager_instance(
        api_keys=["key1", "key2", "key3"],
        vertex_api_keys=["vkey1", "vkey2"]
    )

@pytest.mark.asyncio
async def test_get_next_key(key_manager):
    """Test the get_next_key method."""
    assert await key_manager.get_next_key() == "key1"
    assert await key_manager.get_next_key() == "key2"
    assert await key_manager.get_next_key() == "key3"
    assert await key_manager.get_next_key() == "key1"

@pytest.mark.asyncio
async def test_get_next_vertex_key(key_manager):
    """Test the get_next_vertex_key method."""
    assert await key_manager.get_next_vertex_key() == "vkey1"
    assert await key_manager.get_next_vertex_key() == "vkey2"
    assert await key_manager.get_next_vertex_key() == "vkey1"

@pytest.mark.asyncio
async def test_key_validity(key_manager):
    """Test the is_key_valid and handle_api_failure methods."""
    key_manager.MAX_FAILURES = 2
    assert await key_manager.is_key_valid("key1") is True
    await key_manager.handle_api_failure("key1", "some_model", 1)
    assert await key_manager.is_key_valid("key1") is True
    await key_manager.handle_api_failure("key1", "some_model", 2)
    assert await key_manager.is_key_valid("key1") is False

@pytest.mark.asyncio
async def test_reset_key_failure_count(key_manager):
    """Test resetting a key's failure count."""
    key_manager.MAX_FAILURES = 1
    await key_manager.handle_api_failure("key2", "some_model", 1)
    assert await key_manager.is_key_valid("key2") is False
    await key_manager.reset_key_failure_count("key2")
    assert await key_manager.is_key_valid("key2") is True

@pytest.mark.asyncio
async def test_get_next_working_key_with_usage(key_manager):
    """Test getting the next working key based on RPM."""
    with patch("app.service.key.key_manager_v1.get_usage_stats_by_key_and_model", new_callable=AsyncMock) as mock_get_usage:
        mock_get_usage.side_effect = [
            {"rpd": 100, "rpm": 10, "tpm": 1000, "exhausted": False},  # key1
            {"rpd": 50, "rpm": 5, "tpm": 500, "exhausted": False},   # key2
            {"rpd": 200, "rpm": 20, "tpm": 2000, "exhausted": False},  # key3
        ]
        best_key = await key_manager.get_next_working_key("some_model")
        assert best_key == "key2"

@pytest.mark.asyncio
async def test_get_next_working_key_with_combined_usage(key_manager):
    """Test getting the next working key based on combined RPM, RPD, and TPM."""
    with patch("app.service.key.key_manager_v1.get_usage_stats_by_key_and_model", new_callable=AsyncMock) as mock_get_usage:
        mock_get_usage.side_effect = [
            {"rpd": 100, "rpm": 10, "tpm": 1000, "exhausted": False},  # key1
            {"rpd": 50, "rpm": 10, "tpm": 500, "exhausted": False},   # key2
            {"rpd": 100, "rpm": 10, "tpm": 500, "exhausted": False},  # key3
        ]
        best_key = await key_manager.get_next_working_key("some_model")
        assert best_key == "key2"


@pytest.mark.asyncio
async def test_get_next_working_key_with_exhausted_key(key_manager):
    """Test that exhausted keys are not selected."""
    with patch("app.service.key.key_manager_v1.get_usage_stats_by_key_and_model", new_callable=AsyncMock) as mock_get_usage:
        mock_get_usage.side_effect = [
            {"rpd": 100, "rpm": 10, "tpm": 1000, "exhausted": True, "rpm_timestamp": datetime.now()},  # key1
            {"rpd": 50, "rpm": 5, "tpm": 500, "exhausted": False},   # key2
            {"rpd": 200, "rpm": 20, "tpm": 2000, "exhausted": False},  # key3
        ]
        best_key = await key_manager.get_next_working_key("some_model")
        assert best_key == "key2"

@pytest.mark.asyncio
async def test_get_next_working_key_with_cooldown(key_manager):
    """Test that exhausted keys are re-activated after a cooldown."""
    with patch("app.service.key.key_manager_v1.get_usage_stats_by_key_and_model", new_callable=AsyncMock) as mock_get_usage, \
         patch("app.service.key.key_manager_v1.set_key_exhausted_status", new_callable=AsyncMock) as mock_set_exhausted:
        mock_get_usage.side_effect = [
            {"rpd": 100, "rpm": 10, "tpm": 1000, "exhausted": True, "rpm_timestamp": datetime.now() - timedelta(minutes=2)},  # key1
            {"rpd": 50, "rpm": 5, "tpm": 500, "exhausted": False},   # key2
            {"rpd": 200, "rpm": 20, "tpm": 2000, "exhausted": False},  # key3
        ]
        best_key = await key_manager.get_next_working_key("some_model")
        assert best_key == "key1"
        mock_set_exhausted.assert_called_once_with("key1", "some_model", False)

@pytest.mark.asyncio
async def test_get_next_working_key_no_valid_keys(key_manager):
    """Test get_next_working_key when all keys have failed."""
    key_manager.MAX_FAILURES = 1
    await key_manager.handle_api_failure("key1", "some_model", 1)
    await key_manager.handle_api_failure("key2", "some_model", 1)
    await key_manager.handle_api_failure("key3", "some_model", 1)
    with patch("app.service.key.key_manager_v1.get_usage_stats_by_key_and_model", new_callable=AsyncMock) as mock_get_usage:
        mock_get_usage.return_value = {"rpd": 0, "rpm": 0, "tpm": 0, "exhausted": False}
        # Should cycle and return the next key in order
        assert await key_manager.get_next_working_key("some_model") == "key1"

@pytest.mark.asyncio
async def test_get_next_working_key_no_usage_stats(key_manager):
    """Test get_next_working_key when usage stats are not available."""
    with patch("app.service.key.key_manager_v1.get_usage_stats_by_key_and_model", new_callable=AsyncMock) as mock_get_usage:
        mock_get_usage.return_value = None
        # Should return the key with the lowest index
        assert await key_manager.get_next_working_key("some_model") == "key1"


@pytest.mark.asyncio
async def test_get_next_working_key_with_mixed_usage_stats(key_manager):
    """Test get_next_working_key with some keys having no usage stats."""
    with patch("app.service.key.key_manager_v1.get_usage_stats_by_key_and_model", new_callable=AsyncMock) as mock_get_usage:
        mock_get_usage.side_effect = [
            {"rpd": 100, "rpm": 10, "tpm": 1000, "exhausted": False},  # key1
            None,                                                    # key2
            {"rpd": 200, "rpm": 20, "tpm": 2000, "exhausted": False},  # key3
        ]
        best_key = await key_manager.get_next_working_key("some_model")
        assert best_key == "key2"

@pytest.mark.asyncio
async def test_get_random_valid_key(key_manager):
    """Test getting a random valid key."""
    key_manager.MAX_FAILURES = 1
    await key_manager.handle_api_failure("key1", "some_model", 1)
    await key_manager.handle_api_failure("key3", "some_model", 1)

    valid_keys = []
    for _ in range(10):
        valid_keys.append(await key_manager.get_random_valid_key())

    # After multiple calls, the only valid key should be the one returned
    assert all(key == "key2" for key in valid_keys)

@pytest.mark.asyncio
async def test_get_keys_by_status(key_manager):
    """Test categorizing keys by their validity status."""
    key_manager.MAX_FAILURES = 1
    failed_key = "key1"
    await key_manager.handle_api_failure(failed_key, "some_model", 1)

    status = await key_manager.get_keys_by_status()
    assert failed_key in status["invalid_keys"]
    assert "key2" in status["valid_keys"]
    assert "key3" in status["valid_keys"]

@pytest.mark.asyncio
async def test_get_paid_key(key_manager):
    """Test getting the paid key."""
    key_manager.paid_key = "paid_key_123"
    assert await key_manager.get_paid_key() == "paid_key_123"

@pytest.mark.asyncio
async def test_reset_all_failure_counts(key_manager):
    """Test resetting all failure counts."""
    key_manager.MAX_FAILURES = 1
    await key_manager.handle_api_failure("key1", "some_model", 1)
    await key_manager.handle_api_failure("key2", "some_model", 1)
    assert await key_manager.is_key_valid("key1") is False
    assert await key_manager.is_key_valid("key2") is False

    await key_manager.reset_failure_counts()
    assert await key_manager.is_key_valid("key1") is True
    assert await key_manager.is_key_valid("key2") is True

@pytest.mark.asyncio
async def test_get_all_keys_with_fail_count(key_manager):
    """Test getting all keys with their failure counts."""
    key_manager.MAX_FAILURES = 2
    await key_manager.handle_api_failure("key1", "some_model", 1)
    await key_manager.handle_api_failure("key3", "some_model", 1)
    await key_manager.handle_api_failure("key3", "some_model", 1)

    all_keys = await key_manager.get_all_keys_with_fail_count()

    assert "key1" in all_keys["valid_keys"]
    assert "key2" in all_keys["valid_keys"]
    assert "key3" in all_keys["invalid_keys"]
    assert all_keys["all_keys"]["key1"] == 1
    assert all_keys["all_keys"]["key2"] == 0
    assert all_keys["all_keys"]["key3"] == 2

@pytest.mark.asyncio
async def test_get_first_valid_key(key_manager):
    """Test getting the first valid key."""
    key_manager.MAX_FAILURES = 1
    await key_manager.handle_api_failure("key1", "some_model", 1)
    assert await key_manager.get_first_valid_key() == "key2"
    await key_manager.handle_api_failure("key2", "some_model", 1)
    assert await key_manager.get_first_valid_key() == "key3"
    await key_manager.handle_api_failure("key3", "some_model", 1)
    # All keys are invalid, should return the first key as fallback
    assert await key_manager.get_first_valid_key() == "key1"

@pytest.mark.asyncio
async def test_get_random_valid_key_no_valid_keys(key_manager):
    """Test getting a random valid key when no valid keys are available."""
    key_manager.MAX_FAILURES = 1
    await key_manager.handle_api_failure("key1", "some_model", 1)
    await key_manager.handle_api_failure("key2", "some_model", 1)
    await key_manager.handle_api_failure("key3", "some_model", 1)
    # All keys are invalid, should return the first key as fallback
    assert await key_manager.get_random_valid_key() == "key1"

# Vertex Tests
@pytest.mark.asyncio
async def test_vertex_key_validity(key_manager):
    """Test the is_vertex_key_valid and handle_vertex_api_failure methods."""
    key_manager.MAX_FAILURES = 2
    assert await key_manager.is_vertex_key_valid("vkey1") is True
    await key_manager.handle_vertex_api_failure("vkey1", 1)
    assert await key_manager.is_vertex_key_valid("vkey1") is True
    await key_manager.handle_vertex_api_failure("vkey1", 2)
    assert await key_manager.is_vertex_key_valid("vkey1") is False

@pytest.mark.asyncio
async def test_reset_vertex_key_failure_count(key_manager):
    """Test resetting a vertex key's failure count."""
    key_manager.MAX_FAILURES = 1
    await key_manager.handle_vertex_api_failure("vkey2", 1)
    assert await key_manager.is_vertex_key_valid("vkey2") is False
    await key_manager.reset_vertex_key_failure_count("vkey2")
    assert await key_manager.is_vertex_key_valid("vkey2") is True

@pytest.mark.asyncio
async def test_get_next_working_vertex_key(key_manager):
    """Test getting the next working vertex key."""
    assert await key_manager.get_next_working_vertex_key() == "vkey1"
    key_manager.MAX_FAILURES = 1
    await key_manager.handle_vertex_api_failure("vkey1", 1)
    assert await key_manager.get_next_working_vertex_key() == "vkey2"
    await key_manager.handle_vertex_api_failure("vkey2", 1)
    # All keys failed, should cycle and return first key
    assert await key_manager.get_next_working_vertex_key() == "vkey1"

@pytest.mark.asyncio
async def test_get_vertex_keys_by_status(key_manager):
    """Test categorizing vertex keys by their validity status."""
    key_manager.MAX_FAILURES = 1
    failed_key = "vkey1"
    await key_manager.handle_vertex_api_failure(failed_key, 1)
    status = await key_manager.get_vertex_keys_by_status()
    assert failed_key in status["invalid_keys"]
    assert "vkey2" in status["valid_keys"]

@pytest.mark.asyncio
async def test_reset_all_vertex_failure_counts(key_manager):
    """Test resetting all vertex failure counts."""
    key_manager.MAX_FAILURES = 1
    await key_manager.handle_vertex_api_failure("vkey1", 1)
    await key_manager.handle_vertex_api_failure("vkey2", 1)
    assert await key_manager.is_vertex_key_valid("vkey1") is False
    assert await key_manager.is_vertex_key_valid("vkey2") is False
    await key_manager.reset_vertex_failure_counts()
    assert await key_manager.is_vertex_key_valid("vkey1") is True
    assert await key_manager.is_vertex_key_valid("vkey2") is True

# Singleton and State Preservation Tests
@pytest.mark.asyncio
async def test_singleton_instance():
    """Test that get_key_manager_instance returns a singleton."""
    km1 = await get_key_manager_instance(["key1"], ["vkey1"])
    km2 = await get_key_manager_instance(["key2"], ["vkey2"]) # These keys should be ignored
    assert km1 is km2
    assert km1.api_keys == ["key1"]
    assert km1.vertex_api_keys == ["vkey1"]

@pytest.mark.asyncio
async def test_reset_and_recreate_preserves_failures():
    """Test that failure counts are preserved after reset and recreation."""
    km1 = await get_key_manager_instance(["key1", "key2"], ["vkey1"])
    km1.MAX_FAILURES = 2
    await km1.handle_api_failure("key1", "some_model", 1)
    await km1.handle_vertex_api_failure("vkey1", 1)

    await reset_key_manager_instance()

    # Recreate with a different set of keys, but including the failed ones
    km2 = await get_key_manager_instance(["key1", "key3"], ["vkey1", "vkey3"])
    assert await km2.get_fail_count("key1") == 1
    assert await km2.get_fail_count("key3") == 0
    assert await km2.get_vertex_fail_count("vkey1") == 1
    assert await km2.get_vertex_fail_count("vkey3") == 0
    # Check that keys not in the new list are gone
    assert "key2" not in km2.key_failure_counts

@pytest.mark.asyncio
async def test_reset_and_recreate_preserves_cycle_position():
    """Test that the key cycle position is preserved after reset."""
    km1 = await get_key_manager_instance(["key1", "key2", "key3"], ["vkey1", "vkey2"])
    assert await km1.get_next_key() == "key1"
    assert await km1.get_next_key() == "key2"
    assert await km1.get_next_vertex_key() == "vkey1"

    await reset_key_manager_instance()

    # Recreate, the next key should be key3
    km2 = await get_key_manager_instance(["key1", "key2", "key3"], ["vkey1", "vkey2"])
    assert await km2.get_next_key() == "key3"
    assert await km2.get_next_vertex_key() == "vkey2"

@pytest.mark.asyncio
async def test_initialization_with_empty_keys():
    """Test KeyManager initialization with empty key lists."""
    km = await get_key_manager_instance([], [])
    assert km.api_keys == []
    assert km.vertex_api_keys == []
    assert await km.get_first_valid_key() == ""
    assert await km.get_random_valid_key() == ""
    k = await km.get_next_key()
    assert k == ""
    v = await km.get_next_vertex_key()
    assert v == ""


