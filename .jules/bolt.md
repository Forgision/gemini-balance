## 2024-05-23 - Pandas MultiIndex Performance
**Learning:** `df.index.get_level_values("level_name")` creates a full copy of the index level, leading to O(N) performance for existence checks.
**Action:** Use `df.xs(key, level="level_name")` inside a `try...except KeyError` block for O(1)/O(log N) lookup when checking if a key exists in a specific level of a MultiIndex.
