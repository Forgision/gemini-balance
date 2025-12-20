## 2025-12-20 - Redundant Sorting in Hot Path
**Learning:** `KeyManager._model_normalization` was sorting a list of models O(N log N) on every key request, despite the list being static after initialization. This contradicted the design intent and added unnecessary latency.
**Action:** Always check if a loop iterates over a `sorted()` collection and if that sort can be moved to initialization time, especially in hot paths like request handlers.
