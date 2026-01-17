## 2026-01-17 - Redundant Sorting in Hot Path
**Learning:** The `KeyManager` was re-sorting the `rate_limit_models` list (O(N log N)) on every single API request (`get_key`) and usage update. This was a hidden CPU cost in a high-frequency method.
**Action:** Always check loop invariants in hot paths. For static configuration lists, sort once during initialization and use `functools.lru_cache` for lookups to achieve O(1) performance.
