## 2024-05-22 - Pandas `sort_values` Overhead in Hot Paths
**Learning:** In Pandas, `sort_values()` is O(N log N) and creates a full copy/view. When selecting a single "best" row (e.g., max value), `idxmax()` is O(N) and 18x faster (0.3ms vs 0.017ms for 1000 rows).
**Action:** For "best-of" selection logic in hot paths, always prefer `idxmax()`/`idxmin()` over `sort_values().iloc[0]`.
