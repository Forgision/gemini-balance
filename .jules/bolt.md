## 2025-12-23 - [Performance] Redundant Sorting in Hot Path
**Learning:** `sorted()` is fast but not free. Calling `sorted()` on a list inside a hot path (every request) adds unnecessary `O(N log N)` overhead and list allocation, especially when the list is effectively static or changes infrequently.
**Action:** Pre-sort lists during initialization or update, and iterate over the pre-sorted list in the hot path.
