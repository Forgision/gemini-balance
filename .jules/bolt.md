## 2024-12-17 - Pandas MultiIndex Performance
**Learning:** `df.index.get_level_values("level")` creates a new Index object and scans it, making it O(N). In high-frequency paths like `get_key`, this adds significant overhead.
**Action:** Use `df.xs(key, level=...)` wrapped in `try/except KeyError` for O(1)/O(log N) lookups, or check membership against a pre-computed set if existence check is the only goal.
