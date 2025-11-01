import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch
from app.service.key import key_manager as key_manager_module
from app.service.key.key_manager import get_key_manager_instance, reset_key_manager_instance
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

@pytest_asyncio.fixture
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
    with patch("app.service.key.key_manager.get_usage_stats_by_key_and_model", new_callable=AsyncMock) as mock_get_usage:
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
    with patch("app.service.key.key_manager.get_usage_stats_by_key_and_model", new_callable=AsyncMock) as mock_get_usage:
        mock_get_usage.side_effect = [
            {"rpd": 100, "rpm": 10, "tpm": 1000, "exhausted": False},  # key1
            {"rpd": 50, "rpm": 10, "tpm": 500, "exhausted": False},   # key2
            {"rpd": 100, "rpm": 10, "tpm": 500, "exhausted": False},  # key3
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
async def test_get_next_working_key_with_exhausted_key(key_manager):
    """Test that exhausted keys are not selected."""
    with patch("app.service.key.key_manager.get_usage_stats_by_key_and_model", new_callable=AsyncMock) as mock_get_usage:
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
    with patch("app.service.key.key_manager.get_usage_stats_by_key_and_model", new_callable=AsyncMock) as mock_get_usage, \
         patch("app.service.key.key_manager.set_key_exhausted_status", new_callable=AsyncMock) as mock_set_exhausted:
        mock_get_usage.side_effect = [
            {"rpd": 100, "rpm": 10, "tpm": 1000, "exhausted": True, "rpm_timestamp": datetime.now() - timedelta(minutes=2)},  # key1
            {"rpd": 50, "rpm": 5, "tpm": 500, "exhausted": False},   # key2
            {"rpd": 200, "rpm": 20, "tpm": 2000, "exhausted": False},  # key3
        ]
        best_key = await key_manager.get_next_working_key("some_model")
        assert best_key == "key1"
        mock_set_exhausted.assert_called_once_with("key1", "some_model", False)
