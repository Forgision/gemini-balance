## 2024-12-19 - Pandas DataFrame Optimization
**Learning:** `df.loc[idx, col] = val` has significant overhead. In hot paths (e.g. per-request logic), accessing/updating multiple columns individually adds up (5-6ms for 10 updates).
**Action:** Batch updates by reading the row once (`row = df.loc[idx]`), computing new values in variables, and writing back in one go (`df.loc[idx, [cols]] = [vals]`). This reduced latency by ~2.5x in `KeyManager.update_usage`.
