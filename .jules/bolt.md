## 2024-12-30 - Pandas Hot-Path Optimization
**Learning:** For high-frequency scalar updates in Pandas DataFrames (hot paths), use `df.at[row, col] = val` instead of `df.loc[row, col] = val` for a significant performance gain. Note that `at` raises `KeyError` if the column does not exist, so a prior check (`if col in df.columns`) or fallback to `loc` is required if columns might be missing.
**Action:** When optimizing tight loops or frequent updates in DataFrames, prefer direct scalar access methods like `at` and `iat`.
