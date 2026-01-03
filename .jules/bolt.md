## 2026-01-03 - [Pandas DataFrame Scalar Updates]
**Learning:** `df.loc[idx, col]` is significantly slower (~60%) than `df.at[idx, col]` for scalar updates because `loc` creates Series objects. `at` provides direct access but raises `KeyError` if the column doesn't exist, unlike `loc` which might create it.
**Action:** Use `df.at` for high-frequency counters/stats updates in hot paths, but ensure all columns are pre-initialized in the DataFrame.
