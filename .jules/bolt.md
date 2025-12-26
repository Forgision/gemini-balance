## 2024-05-23 - KeyManager DataFrame Optimization
**Learning:** Using `idxmax()` on a Pandas Series is significantly faster (O(N) vs O(N log N)) than sorting the entire DataFrame when you only need the maximum value. Also, `df.at[row, col]` provides O(1) scalar access which is much faster than `df.loc[row, col]` for single value updates in hot paths, avoiding overhead of slicing and index checks.
**Action:** Prefer `idxmax/idxmin` for finding best candidates. Use `at/iat` for single-cell updates in high-frequency loops.
