## 2024-05-23 - Repeated Sort in Hot Path
**Learning:** `KeyManager._model_normalization` was sorting `rate_limit_models` on every call to find the longest matching prefix. This caused unnecessary CPU overhead on every request.
**Action:** Move invariant sorting logic to `__init__` so it runs once. Ensure the data structure remains sorted if updated.
