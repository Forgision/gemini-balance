## 2024-12-31 - [Pandas MultiIndex Performance]
**Learning:** Checking for existence in a Pandas MultiIndex using `df.index.get_level_values(...)` performs a full scan (O(N)). The preferred pattern for partial MultiIndex lookups in this codebase is to use `df.xs(..., drop_level=False)` or `df.loc[...]` inside a `try/except KeyError` block.
**Action:** Use EAFP with `df.xs` for indexed lookups instead of checking membership in `get_level_values`.

## 2024-12-31 - [Pandas Selection Performance]
**Learning:** When selecting a single "best" row based on a column maximum, `df[col].idxmax()` (O(M)) is significantly faster than `df.sort_values(by=col).iloc[0]` (O(M log M)) because it avoids sorting the entire dataframe/series.
**Action:** Use `idxmax()` when only the top/bottom element is needed.
