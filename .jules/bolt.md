## 2024-05-24 - Pandas DataFrame Optimization
**Learning:** Pandas `loc` is significantly slower (5-10x) than `at` for scalar value updates, especially with MultiIndex. However, `at` raises `KeyError` if the column does not exist, whereas `loc` creates it.
**Action:** Use `at` for hot-path updates to known columns. For columns that might be missing, check existence first: `if col in df.columns: df.at[...] else: df.loc[...]`. This balances performance with safety.
