## 2025-12-21 - Python Sorting Overhead
**Learning:** `sorted()` creates a new list copy every time it's called. Calling it inside a loop or hot path (like model normalization per request) adds unnecessary O(N log N) overhead and allocation.
**Action:** Pre-sort lists during initialization if the order is static, and iterate over the pre-sorted list in hot paths.
