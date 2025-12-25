## 2024-05-23 - KeyManager Selection Optimization
**Learning:** `pandas.DataFrame.sort_values` is $O(N \log N)$ and copies data, which is expensive for frequent lookups. `Series.idxmax()` is $O(N)$ and optimal for finding a single best candidate.
**Action:** When finding the "best" row based on a single metric, prefer `idxmax/idxmin` or `nlargest(1)` over full dataframe sorting. Also, use EAFP (try/except) instead of `index.get_level_values(...)` check (which scans the full index) for looking up keys in MultiIndex.
