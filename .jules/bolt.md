## 2024-01-05 - Optimizing Pandas for High-Throughput Key Selection
**Learning:** `df.sort_values` is O(N log N) which is expensive in hot paths. `idxmax()` is O(N) and significantly faster for finding a single top candidate. Also, `df.index.get_level_values(...)` performs a full index scan (O(N)), which is inefficient for existence checks compared to `try...except KeyError` on `df.xs` (O(1) average).
**Action:** Use `idxmax()` instead of `sort_values(...).iloc[0]` when only the single best record is needed. Prefer EAFP (try/except) for MultiIndex lookups to avoid expensive index scans.
