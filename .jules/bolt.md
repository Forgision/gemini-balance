## 2025-12-28 - Optimizing Pandas MultiIndex Lookup
**Learning:** Checking for existence in a Pandas MultiIndex using `df.index.get_level_values(...)` performs a full scan (O(N)). The preferred pattern for partial MultiIndex lookups in this codebase is to use `df.xs(..., drop_level=False)` or `df.loc[...]` inside a `try/except KeyError` block (EAFP).
**Action:** Replace inefficient existence checks with direct try/except access patterns.
