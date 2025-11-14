# KeyManager v2 Implementation Plan

> **⚠️ STATUS: SUPERSEDED**  
> **This document has been superseded by `FIX_PLAN.md`**  
> **Please refer to `FIX_PLAN.md` for the current implementation plan**  
> **This document is kept for historical reference only**

## Overview
This document was an earlier implementation plan. It has been superseded by `FIX_PLAN.md` which contains:
- Detailed fixes for 30 critical issues identified in deep assessment
- Updated design decisions based on user feedback
- Minimal v2 core logic changes approach
- Comprehensive implementation steps

**Current Implementation Guide**: See `FIX_PLAN.md`  
**Architecture Reference**: See `MIGRATION_AUDIT.md`

---

## Historical Notes

The following sections contain the original plan. Many decisions have been updated:
- ✅ Singleton pattern → Changed to `app.state` approach
- ✅ Separate methods → Changed to unified methods with `is_vertex_key` parameter
- ✅ Manual reset methods → Changed to use `reset_key_failure_count()` adapter
- ✅ Status format → Changed to v2 format with UI updates

**Please refer to `FIX_PLAN.md` for current implementation details.**

## Required Implementations

### 1. ✅ `get_paid_key()` Method
**Status**: To Implement  
**Priority**: HIGH  
**Description**: Simple method to return the paid API key from settings.

**Implementation**:
```python
async def get_paid_key(self) -> str:
    """Get the paid API key for premium features like image generation."""
    from app.config.config import settings
    return settings.PAID_KEY
```

**Usage**: Used in `app/router/openai_routes.py:90` for image chat functionality.

---

### 2. ✅ `get_next_key()` and `get_next_vertex_key()` Methods
**Status**: To Implement  
**Priority**: HIGH  
**Description**: Simple round-robin key selection methods.

**Implementation**:
```python
async def get_next_key(self) -> str:
    """Get the next API key in round-robin fashion."""
    try:
        return next(self.api_keys_cycle)
    except StopIteration:
        logger.warning("API key cycle is empty.")
        return ""

async def get_next_vertex_key(self) -> str:
    """Get the next Vertex API key in round-robin fashion."""
    try:
        return next(self.vertex_api_keys_cycle)
    except StopIteration:
        logger.warning("Vertex API key cycle is empty.")
        return ""
```

**Usage**: 
- `app/service/files/files_service.py:65`
- Singleton pattern state preservation
- Fallback mechanism in `get_key()` when model not found

---

### 3. ⏸️ Replace `get_next_working_key` with `get_key`
**Status**: Pending Permission (After all implementations)  
**Priority**: MEDIUM  
**Description**: Replace all usages of `get_next_working_key()` and `get_next_working_vertex_key()` with `get_key()` method.

**Files to Update**:
- `app/router/openai_routes.py`
- `app/router/gemini_routes.py`
- `app/router/openai_compatible_routes.py`
- `app/router/vertex_express_routes.py`
- `app/service/claude_proxy_service.py`

**Note**: This will be done after all required methods are implemented and tested.

---

### 4. ✅ `get_usage_stats()` Method
**Status**: To Implement  
**Priority**: MEDIUM  
**Description**: Return usage statistics for a specific key and model.

**Implementation**:
```python
async def get_usage_stats(
    self, api_key: str, model_name: str
) -> Optional[Dict[str, Any]]:
    """Get usage statistics for a given key and model."""
    if not self.is_ready:
        await self._check_ready()
    
    match, model_name = self._model_normalization(model_name)
    
    async with self.lock:
        try:
            # Determine if vertex key
            is_vertex_key = api_key in self.vertex_api_keys
            idx = (model_name, is_vertex_key, api_key)
            
            if idx not in self.df.index:
                return None
            
            row = self.df.loc[idx]
            return {
                "rpm": int(row["rpm"]),
                "tpm": int(row["tpm"]),
                "rpd": int(row["rpd"]),
                "max_rpm": int(row["max_rpm"]),
                "max_tpm": int(row["max_tpm"]),
                "max_rpd": int(row["max_rpd"]),
                "rpm_left": int(row.get("rpm_left", 0)),
                "tpm_left": int(row.get("tpm_left", 0)),
                "rpd_left": int(row.get("rpd_left", 0)),
                "is_active": bool(row["is_active"]),
                "is_exhausted": bool(row["is_exhausted"]),
                "last_used": row["last_used"],
                "minute_reset_time": row.get("minute_reset_time"),
                "day_reset_time": row.get("day_reset_time"),
            }
        except (KeyError, IndexError):
            return None
```

**Usage**: Used internally by `get_next_working_key()` in v1, and for monitoring/debugging.

---

### 5. ⏸️ Audit `update_usage()` for Failure Handling
**Status**: Pending Audit (After implementation)  
**Priority**: MEDIUM  
**Description**: Verify that `update_usage()` with `error=True` and `error_type` parameters can replace `handle_api_failure()` and `handle_vertex_api_failure()`.

**Current `update_usage()` Capabilities**:
- ✅ Handles `error_type="permanent"` - deactivates key for all models
- ✅ Handles `error_type="429"` - sets exhausted flag for specific model
- ✅ Updates usage metrics on success

**Comparison with v1 Methods**:
- v1 `handle_api_failure()`: Increments failure count, sets exhausted status, returns new key
- v2 `update_usage()`: Sets exhausted/active flags directly, no failure count tracking

**Recommendation**: `update_usage()` is sufficient for error handling. The retry logic in `retry_handler.py` should handle getting a new key after failure.

**Action Required**: After implementation, audit all call sites to ensure they use `update_usage()` correctly.

---

### 6. ⏸️ Audit `reset_usage_bg()` for Reset Functionality
**Status**: Pending Audit (After implementation)  
**Priority**: LOW  
**Description**: Verify that `reset_usage_bg()` background task is sufficient for all reset needs.

**Current `reset_usage_bg()` Capabilities**:
- ✅ Automatically resets RPM/TPM every minute
- ✅ Automatically resets RPD daily
- ✅ Resets exhausted flags on minute reset
- ✅ Commits to database after resets

**Comparison with v1 Methods**:
- v1 has manual reset methods: `reset_failure_counts()`, `reset_vertex_failure_counts()`, `reset_key_failure_count()`, `reset_vertex_key_failure_count()`
- v2 has automatic background resets via `reset_usage_bg()`

**Recommendation**: `reset_usage_bg()` is sufficient. Manual reset methods are not needed because:
1. Rate limits reset automatically
2. Exhausted flags reset automatically
3. No failure count tracking (uses is_active/is_exhausted instead)

**Action Required**: After implementation, verify no manual reset functionality is needed.

---

### 7. ✅ `get_state()` Method
**Status**: To Implement  
**Priority**: HIGH  
**Description**: Comprehensive state method that returns table data for monitoring page.

**Implementation**:
```python
async def get_state(self) -> Dict[str, Any]:
    """
    Get comprehensive state of all keys for monitoring.
    Returns data structured for table display showing available and exhausted keys per model.
    """
    if not self.is_ready:
        await self._check_ready()
    
    async with self.lock:
        # Create a copy to avoid holding lock during processing
        df_copy = self.df.copy()
    
    # Group by model and vertex key type
    state = {
        "models": {},
        "summary": {
            "total_keys": len(self.api_keys) + len(self.vertex_api_keys),
            "total_models": len(self.rate_limit_models),
        }
    }
    
    for model in self.rate_limit_models:
        if model not in df_copy.index.get_level_values("model_name"):
            continue
        
        model_data = {
            "model_name": model,
            "regular_keys": {
                "available": [],
                "exhausted": [],
                "inactive": [],
            },
            "vertex_keys": {
                "available": [],
                "exhausted": [],
                "inactive": [],
            }
        }
        
        # Process regular keys
        try:
            regular_df = df_copy.xs((model, False), level=["model_name", "is_vertex_key"])
            for api_key, row in regular_df.iterrows():
                key_info = {
                    "api_key": api_key,
                    "rpm": int(row["rpm"]),
                    "tpm": int(row["tpm"]),
                    "rpd": int(row["rpd"]),
                    "max_rpm": int(row["max_rpm"]),
                    "max_tpm": int(row["max_tpm"]),
                    "max_rpd": int(row["max_rpd"]),
                    "rpm_left": int(row.get("rpm_left", 0)),
                    "tpm_left": int(row.get("tpm_left", 0)),
                    "rpd_left": int(row.get("rpd_left", 0)),
                    "last_used": row["last_used"].isoformat() if pd.notna(row["last_used"]) else None,
                }
                
                if not row["is_active"]:
                    model_data["regular_keys"]["inactive"].append(key_info)
                elif row["is_exhausted"]:
                    model_data["regular_keys"]["exhausted"].append(key_info)
                else:
                    model_data["regular_keys"]["available"].append(key_info)
        except KeyError:
            pass
        
        # Process vertex keys
        try:
            vertex_df = df_copy.xs((model, True), level=["model_name", "is_vertex_key"])
            for api_key, row in vertex_df.iterrows():
                key_info = {
                    "api_key": api_key,
                    "rpm": int(row["rpm"]),
                    "tpm": int(row["tpm"]),
                    "rpd": int(row["rpd"]),
                    "max_rpm": int(row["max_rpm"]),
                    "max_tpm": int(row["max_tpm"]),
                    "max_rpd": int(row["max_rpd"]),
                    "rpm_left": int(row.get("rpm_left", 0)),
                    "tpm_left": int(row.get("tpm_left", 0)),
                    "rpd_left": int(row.get("rpd_left", 0)),
                    "last_used": row["last_used"].isoformat() if pd.notna(row["last_used"]) else None,
                }
                
                if not row["is_active"]:
                    model_data["vertex_keys"]["inactive"].append(key_info)
                elif row["is_exhausted"]:
                    model_data["vertex_keys"]["exhausted"].append(key_info)
                else:
                    model_data["vertex_keys"]["available"].append(key_info)
        except KeyError:
            pass
        
        state["models"][model] = model_data
    
    return state
```

**Usage**: Replace `get_keys_by_status()` and `get_all_keys_with_fail_count()` in monitoring endpoints.

**Return Format**:
```json
{
  "models": {
    "gemini-pro": {
      "model_name": "gemini-pro",
      "regular_keys": {
        "available": [{"api_key": "...", "rpm": 10, ...}],
        "exhausted": [{"api_key": "...", "rpm": 15, ...}],
        "inactive": []
      },
      "vertex_keys": {...}
    }
  },
  "summary": {
    "total_keys": 10,
    "total_models": 5
  }
}
```

---

### 8. ❌ Utility Methods - Not Needed
**Status**: Skipped  
**Description**: Methods like `get_first_valid_key()`, `get_random_valid_key()` are not needed as `get_key()` handles key selection intelligently.

---

### 9. ✅ Covered in #7
**Status**: Covered  
**Description**: Status methods are replaced by `get_state()`.

---

### 10. ✅ Singleton Pattern with Locks
**Status**: To Implement  
**Priority**: CRITICAL  
**Description**: Implement singleton pattern using asyncio.Lock (better approach than v1's complex state preservation).

**Implementation**:
```python
# At module level
_singleton_instance: Optional[KeyManager] = None
_singleton_lock = asyncio.Lock()

async def get_key_manager_instance(
    api_keys: Optional[list[str]] = None,
    vertex_api_keys: Optional[list[str]] = None,
) -> KeyManager:
    """
    Get the KeyManager singleton instance.
    Uses asyncio.Lock for thread-safe singleton creation.
    """
    global _singleton_instance
    
    async with _singleton_lock:
        if _singleton_instance is None:
            if api_keys is None or vertex_api_keys is None:
                raise ValueError(
                    "API keys and vertex API keys are required for first initialization."
                )
            
            # Create instance
            _singleton_instance = KeyManager(
                api_keys=api_keys,
                vertex_api_keys=vertex_api_keys,
                async_session_maker=AsyncSessionLocal,
            )
            
            # Initialize
            success = await _singleton_instance.init()
            if not success:
                _singleton_instance = None
                raise RuntimeError("Failed to initialize KeyManager")
            
            logger.info(
                f"KeyManager singleton created with {len(api_keys)} API keys "
                f"and {len(vertex_api_keys)} Vertex keys."
            )
        
        return _singleton_instance

async def reset_key_manager_instance():
    """
    Reset the KeyManager singleton instance.
    Gracefully shuts down the current instance and clears the singleton.
    """
    global _singleton_instance
    
    async with _singleton_lock:
        if _singleton_instance is not None:
            await _singleton_instance.shutdown()
            _singleton_instance = None
            logger.info("KeyManager singleton reset.")
```

**Advantages over v1**:
- ✅ Simpler: No complex state preservation
- ✅ Thread-safe: Uses asyncio.Lock
- ✅ Clean: Proper shutdown handling
- ✅ Better: Locks are more appropriate for async code

**Usage**: Replace all calls to v1's `get_key_manager_instance()`.

---

## Missing Attributes

### ✅ All Required Attributes Present
The v2 class already has all required attributes:
- ✅ `self.api_keys` and `self.vertex_api_keys`
- ✅ `self.api_keys_cycle` and `self.vertex_api_keys_cycle`
- ✅ `self.lock` (asyncio.Lock)
- ✅ `self.is_ready` flag
- ✅ `self.df` DataFrame with all state

**Note**: v2 doesn't need `MAX_FAILURES` or `paid_key` as instance attributes since they can be accessed from settings when needed.

---

## Implementation Order

1. ✅ **Phase 1: Core Methods** (Immediate)
   - Implement `get_paid_key()`
   - Implement `get_next_key()` and `get_next_vertex_key()`
   - Implement `get_usage_stats()`
   - Implement `get_state()`

2. ✅ **Phase 2: Singleton Pattern** (Immediate)
   - Implement `get_key_manager_instance()`
   - Implement `reset_key_manager_instance()`
   - Update `app/dependencies.py` to use v2

3. ⏸️ **Phase 3: Testing & Auditing** (After Phase 1 & 2)
   - Test all new methods
   - Audit `update_usage()` for failure handling
   - Audit `reset_usage_bg()` for reset functionality
   - Verify backward compatibility

4. ⏸️ **Phase 4: Migration** (After Permission)
   - Replace `get_next_working_key` with `get_key` (with permission)
   - Update monitoring endpoints to use `get_state()`
   - Update error handling to use `update_usage()`

---

## Testing Requirements

- [ ] Test `get_paid_key()` returns correct value
- [ ] Test `get_next_key()` and `get_next_vertex_key()` round-robin behavior
- [ ] Test `get_usage_stats()` returns correct data
- [ ] Test `get_state()` returns comprehensive state
- [ ] Test singleton pattern thread-safety
- [ ] Test singleton reset functionality
- [ ] Test backward compatibility with existing code

---

## Notes

- The v2 architecture is superior: uses DataFrame for state management, automatic resets, and better error handling
- No failure count tracking needed: `is_active`/`is_exhausted` flags are more reliable
- Background reset task eliminates need for manual reset methods
- `get_state()` provides more informative monitoring than separate status methods

