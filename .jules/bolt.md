## 2024-12-18 - Pandas MultiIndex Performance
**Learning:** `get_level_values` on a Pandas MultiIndex creates a full index copy and scans linearly. Using `.loc[(level0, level1, slice(None))]` is ~35% faster for lookups.
**Action:** Use `df.loc` slicing for MultiIndex lookups instead of chained `.xs()` or `get_level_values()` checks.
