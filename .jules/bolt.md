## 2024-02-14 - Pandas Scalar Update Optimization
**Learning:** Using `df.loc[row_indexer, col_indexer]` to update a scalar value in a Pandas DataFrame (e.g., `df.loc[idx, "col"] = val`) involves significant overhead because it may create intermediate Series objects.
**Action:** For high-frequency single-value updates, always use `df.at[row_indexer, col_indexer] = val`. It provides direct access to the scalar value and is significantly faster (O(1)). Also, avoiding `row = df.loc[idx, :]` when only specific columns are needed prevents expensive Series creation.
