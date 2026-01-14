## 2024-05-22 - Pandas MultiIndex & Scalar Performance
**Learning:** Checking for existence in a Pandas MultiIndex using `df.index.get_level_values(...)` performs a full scan (O(N)). The preferred pattern for partial MultiIndex lookups in this codebase is to use `df.xs(..., drop_level=False)` or `df.loc[...]` inside a `try/except KeyError` block. Additionally, `df.at[]` provides a ~5x speedup over `df.loc[]` for scalar access.
**Action:** Use EAFP with `xs` for lookups and `at` for scalar updates in hot paths like `KeyManager`.
