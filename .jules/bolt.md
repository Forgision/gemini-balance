## 2024-10-18 - Pandas Index Scans in Hot Paths
**Learning:** `model_name in df.index.get_level_values(...)` creates a full index scan ($O(N)$) and object allocation. Using `df.xs()` with `try/except KeyError` is significantly faster ($O(1)$ or $O(\log N)$) and avoids redundant checks.
**Action:** When working with Pandas MultiIndex, prefer exception handling for existence checks over explicit membership tests on `get_level_values()`.
