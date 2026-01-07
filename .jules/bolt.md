## 2024-03-24 - Pandas MultiIndex Performance
**Learning:** Checking existence in a Pandas MultiIndex using `df.index.get_level_values(...)` performs a full scan (O(N)).
**Action:** The preferred pattern for partial MultiIndex lookups in this codebase is to use `df.xs(..., drop_level=False)` or `df.loc[...]` inside a `try/except KeyError` block (O(1)).
