from app.service.key.key_manager import KeyManager
from unittest.mock import MagicMock

class TestKeyManagerPerf:
    def test_model_normalization_sorting_behavior(self):
        """
        Verify that rate_limit_models is sorted in __init__ and
        _model_normalization uses the pre-sorted list.
        """
        rate_limit_data = {
            "gemini-1.5": {},
            "gemini-1.5-flash": {},
            "gemini": {}
        }

        # Instantiate KeyManager
        # We mock dependencies to avoid side effects
        km = KeyManager([], [], MagicMock(), rate_limit_data=rate_limit_data)

        # Check if rate_limit_models is sorted by length (descending) immediately after init
        # This asserts that the fix in __init__ works
        expected_order = ["gemini-1.5-flash", "gemini-1.5", "gemini"]
        assert km.rate_limit_models == expected_order

        # Check normalization logic
        # This asserts that the fix in _model_normalization works (correct logic without sorted())
        assert km._model_normalization("gemini-1.5-flash-001") == "gemini-1.5-flash"
        assert km._model_normalization("gemini-1.5-pro") == "gemini-1.5"
        assert km._model_normalization("gemini-something") == "gemini"

        # Verify correctness with unsorted input
        unsorted_data = {
            "a": {},
            "aaa": {},
            "aa": {}
        }
        km2 = KeyManager([], [], MagicMock(), rate_limit_data=unsorted_data)
        assert km2.rate_limit_models == ["aaa", "aa", "a"]
        assert km2._model_normalization("aaaa") == "aaa"
