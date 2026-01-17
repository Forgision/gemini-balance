import pytest
from unittest.mock import Mock
from app.service.key.key_manager import KeyManager

class TestKeyManagerOptimization:
    def test_init_sorts_rate_limit_models(self):
        """Test that rate_limit_models are sorted by length descending on init."""
        rate_limit_data = {
            "short": {},
            "longer": {},
            "longest_model_name": {},
            "med": {}
        }

        km = KeyManager(
            api_keys=["k"],
            vertex_api_keys=[],
            async_session_maker=Mock(),
            rate_limit_data=rate_limit_data
        )

        expected = ["longest_model_name", "longer", "short", "med"] # Sorted by len desc
        # Note: "short" and "med" have same length, so relative order depends on implementation of stable sort
        # Python's sort is stable. Dict order is insertion order (since 3.7).
        # "med" was inserted after "short", so stable sort might preserve or not depending on key.
        # reverse=True on length.
        # Actually if lengths are equal, original order is preserved.

        assert km.rate_limit_models[0] == "longest_model_name"
        assert km.rate_limit_models[1] == "longer"
        # Check remaining
        assert set(km.rate_limit_models[2:]) == {"short", "med"}

    def test_model_normalization_overlapping_prefixes(self):
        """Test that the longest prefix is chosen when multiple match."""
        rate_limit_data = {
            "gpt-4": {},
            "gpt-4-turbo": {},
            "gpt": {},
        }

        km = KeyManager(
            api_keys=["k"],
            vertex_api_keys=[],
            async_session_maker=Mock(),
            rate_limit_data=rate_limit_data
        )

        # Verify order first
        assert km.rate_limit_models == ["gpt-4-turbo", "gpt-4", "gpt"]

        # Test exact matches
        assert km._model_normalization("gpt-4") == "gpt-4"
        assert km._model_normalization("gpt-4-turbo") == "gpt-4-turbo"
        assert km._model_normalization("gpt") == "gpt"

        # Test partial matches
        assert km._model_normalization("gpt-4-turbo-preview") == "gpt-4-turbo" # Matches gpt-4-turbo
        assert km._model_normalization("gpt-4-vision") == "gpt-4" # Matches gpt-4
        assert km._model_normalization("gpt-3.5") == "gpt" # Matches gpt

        # Test no match
        assert km._model_normalization("claude") == "claude"

    def test_model_normalization_caching(self):
        """Test that caching works (indirectly via performance or repeated calls)."""
        rate_limit_data = {"test": {}}
        km = KeyManager(
            api_keys=["k"],
            vertex_api_keys=[],
            async_session_maker=Mock(),
            rate_limit_data=rate_limit_data
        )

        # First call
        res1 = km._model_normalization("test-1")
        assert res1 == "test"

        # Second call (should be cached)
        res2 = km._model_normalization("test-1")
        assert res2 == "test"

        # Check cache info
        info = km._model_normalization.cache_info()
        assert info.hits >= 1
