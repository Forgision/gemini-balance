# Bolt Journal

## 2024-03-22 - O(N) Pandas Lookup
**Learning:** `df.index.get_level_values(...)` creates a new Index object and performs a full scan, which is O(N) where N is the number of rows. In high-frequency paths (like `get_key`), checking existence via `try/except KeyError` on `df.xs` is significantly faster (O(1) or O(log N)).
**Action:** Prefer EAFP (Easier to Ask for Forgiveness than Permission) when checking for existence in Pandas MultiIndex, especially for partial key lookups.
