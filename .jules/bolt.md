## 2026-01-15 - Pandas Scalar Access Performance
**Learning:** Pandas `df.loc[idx, col]` is significantly slower (~5-6x) than `df.at[idx, col]` for scalar access/updates in high-frequency paths. Additionally, avoiding full DataFrame column operations (like `df['col'] = ...`) in favor of scalar updates (`df.at[idx, 'col'] = ...`) for single-row updates prevents O(N) complexity per request.
**Action:** Use `df.at` for all single-value reads/writes in critical loops and hot paths.
