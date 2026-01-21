from unittest.mock import MagicMock
from app.service.key.key_manager import KeyManager

class TestKeyManagerOptimization:

    def test_initialization_sorts_models(self):
        """Test that __init__ sorts the rate limit models by length descending."""
        rate_limit_data = {
            "gemini": {},
            "gemini-1.5-flash": {},
            "gemini-1.5": {},
        }

        # Instantiate KeyManager
        km = KeyManager(
            api_keys=["key1"],
            vertex_api_keys=[],
            async_session_maker=MagicMock(),
            rate_limit_data=rate_limit_data
        )

        # Verify it is sorted by length descending
        # NOTE: This test is expected to fail BEFORE the fix because __init__ currently doesn't sort.
        # It relies on init() to sort.
        expected = ["gemini-1.5-flash", "gemini-1.5", "gemini"]
        assert km.rate_limit_models == expected

    def test_model_normalization_logic(self):
        """Test that _model_normalization logic remains correct."""
        rate_limit_data = {
            "gemini": {},
            "gemini-1.5-flash": {},
            "gemini-1.5": {},
        }

        km = KeyManager(
            api_keys=["key1"],
            vertex_api_keys=[],
            async_session_maker=MagicMock(),
            rate_limit_data=rate_limit_data
        )

        # Setup expected state manually if __init__ doesn't do it (simulating post-init state)
        km.rate_limit_models = sorted(list(rate_limit_data.keys()), key=len, reverse=True)

        # Test cases
        assert km._model_normalization("gemini-1.5-flash-001") == "gemini-1.5-flash"
        assert km._model_normalization("gemini-1.5-pro") == "gemini-1.5"
        assert km._model_normalization("gemini-something") == "gemini"
        assert km._model_normalization("unknown") == "unknown"
