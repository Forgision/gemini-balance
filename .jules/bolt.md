## 2025-12-14 - Pandas DataFrame Update Performance
**Learning:** Updating a single row in a pandas DataFrame should avoid full-column recalculations. `df['col'] = ...` recalculates the entire column, which is O(N).
**Action:** Use `df.loc[idx, 'col'] = val` for atomic updates and calculate derived values for that row specifically.
