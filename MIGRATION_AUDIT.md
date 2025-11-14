# KeyManager v1 → v2 Migration Audit & Assessment

> **Status**: ✅ **REFERENCE DOCUMENT**  
> **Purpose**: Architecture comparison, design decisions, and migration strategy overview  
> **Related Documents**: See `FIX_PLAN.md` for detailed implementation steps

## Executive Summary

This document provides a comprehensive audit of the differences between `key_manager.py` (v1) and `key_manager_v2.py` (v2), identifies all usage patterns, and provides the migration strategy overview.

**For detailed implementation steps and code fixes**, see `FIX_PLAN.md` (primary implementation guide).

---

## 1. Architecture Comparison

### v1 Architecture
- **State Management**: In-memory dictionaries (`key_failure_counts`, `vertex_key_failure_counts`)
- **Rate Limiting**: External database (`get_usage_stats_by_key_and_model`, `set_key_exhausted_status`)
- **Key Selection**: Multiple methods with different strategies
- **Error Handling**: Failure count tracking with MAX_FAILURES threshold
- **Resets**: Manual reset methods required

### v2 Architecture
- **State Management**: Pandas DataFrame with multi-index (model_name, is_vertex_key, api_key)
- **Rate Limiting**: Built-in SQLAlchemy database + in-memory DataFrame
- **Key Selection**: Single intelligent method (`get_key()`) that selects best key
- **Error Handling**: Direct flag-based (`is_active`, `is_exhausted`) - no failure counts
- **Resets**: Automatic background task (`reset_usage_bg()`) handles all resets

### Key Advantages of v2
1. ✅ **Unified State**: All state in one DataFrame, easier to query and maintain
2. ✅ **Automatic Resets**: No manual intervention needed
3. ✅ **Better Key Selection**: Considers TPM left, not just round-robin
4. ✅ **Model-Aware**: Tracks usage per model, not just per key
5. ✅ **Thread-Safe**: Uses asyncio.Lock for all operations

---

## 2. Method-by-Method Analysis

### 2.1 Key Retrieval Methods

#### `get_paid_key() -> str`
- **v1**: ✅ Exists - returns `self.paid_key` (from settings)
- **v2**: ❌ Missing
- **Usage**: `app/router/openai_routes.py:90` (image chat)
- **Migration**: Simple - just return `settings.PAID_KEY`

#### `get_next_key(is_vertex_key: bool = False) -> str` ⚠️ **UPDATED DESIGN**
- **v1**: Two separate methods: `get_next_key()` and `get_next_vertex_key()`
- **v2**: ❌ Missing (but has cycles available)
- **Usage**: 
  - `app/service/files/files_service.py:65` (uses `get_next_key()`)
  - Fallback in v2's `get_key()` when model not found
- **Migration**: **Combine into one method** with `is_vertex_key` parameter (like `get_key()`)
  - If `is_vertex_key=True`: use `self.vertex_api_keys_cycle`
  - If `is_vertex_key=False`: use `self.api_keys_cycle`
- **Design Decision**: ✅ Unified API - consistent with `get_key()` method signature

#### `get_next_working_key(model_name: str) -> str`
- **v1**: ✅ Exists - selects key with lowest RPM/RPD/TPM from usage stats
- **v2**: ❌ Missing (but `get_key()` does similar, better logic)
- **Usage**: 
  - `app/router/openai_routes.py:37, 170`
  - `app/router/gemini_routes.py:44`
  - `app/router/openai_compatible_routes.py:28, 141`
  - `app/service/claude_proxy_service.py:316`
- **Migration**: Replace with `get_key(model_name, is_vertex_key=False)`
- **Note**: v2's `get_key()` is superior - selects by TPM left, filters exhausted keys

#### `get_next_working_vertex_key() -> str`
- **v1**: ✅ Exists - finds first valid vertex key
- **v2**: ❌ Missing (but `get_key()` handles this)
- **Usage**: `app/router/vertex_express_routes.py:27`
- **Migration**: Replace with `get_key(model_name, is_vertex_key=True)`

---

### 2.2 Error Handling Methods

#### `handle_api_failure(api_key, model_name, retries, is_vertex_key=False, status_code=None) -> str` ⚠️ **UPDATED DESIGN**
- **v1 Behavior**:
  1. Increments `key_failure_counts[api_key]`
  2. If count >= MAX_FAILURES: sets exhausted status in DB
  3. If retries < MAX_RETRIES: returns `get_random_valid_key()`
  4. Else: returns empty string
- **v2 Design**: **Unified method** that:
  1. Calls `update_usage()` with appropriate error_type
  2. Then calls `get_key()` to return new key for retry
  3. Works for both regular and vertex keys via `is_vertex_key` parameter
- **Usage**: 
  - `app/handler/retry_handler.py:53`
  - `app/service/chat/openai_chat_service.py:564`
  - `app/service/chat/gemini_chat_service.py:553`
  - `app/service/chat/vertex_express_chat_service.py:372`
  - `app/service/openai_compatiable/openai_compatiable_service.py:163`
  - `app/service/claude_proxy_service.py:336`
- **Migration Strategy**: 
  - **Single unified method** replaces both `handle_api_failure()` and `handle_vertex_api_failure()`
  - Determines error_type from status_code (429 = rate limit, others = permanent)
  - Calls `update_usage()` then `get_key()` to return new key
  - **Advantage**: Avoids code duplication, consistent API

---

### 2.3 Status & Monitoring Methods

#### `get_keys_by_status() -> dict`
- **v1 Returns**: `{"valid_keys": {key: fail_count}, "invalid_keys": {key: fail_count}}`
- **v2**: ❌ Missing
- **Usage**: 
  - `app/router/routes.py:115`
  - `app/router/gemini_routes.py:352`
  - `app/router/openai_routes.py:189`
- **Migration**: Use new `get_state()` method (see below)

#### `get_all_keys_with_fail_count() -> dict`
- **v1 Returns**: `{"valid_keys": {...}, "invalid_keys": {...}, "all_keys": {...}}`
- **v2**: ❌ Missing
- **Usage**: 
  - `app/router/key_routes.py:29, 84`
- **Migration**: Use new `get_state()` method

#### `get_vertex_keys_by_status() -> dict`
- **v1**: ✅ Exists
- **v2**: ❌ Missing
- **Usage**: Not found in codebase (may be unused)
- **Migration**: Use new `get_state()` method

#### `get_state() -> dict` (NEW - to be implemented) ⚠️ **UPDATED REQUIREMENTS**
- **Purpose**: Comprehensive state for monitoring tables with expandable UI
- **Backend Behavior**: Returns **ALL models** in the data structure
- **UI Behavior** (Frontend responsibility):
  - Display only **3 models initially**: `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.0-flash`
  - Expandable on click (similar to `keyUsageDetailsModal`) to show all models
  - Shows detailed key statistics per model
- **Should Return**: 
  ```python
  {
    "models": {
      "gemini-2.5-pro": {
        "model_name": "gemini-2.5-pro",
        "regular_keys": {
          "available": [
            {
              "api_key": "...",
              "rpm": 10, "tpm": 1000, "rpd": 50,
              "max_rpm": 15, "max_tpm": 1000000, "max_rpd": 1500,
              "rpm_left": 5, "tpm_left": 999000, "rpd_left": 1450,
              "last_used": "2024-01-01T12:00:00Z"
            }
          ],
          "exhausted": [...],
          "inactive": []
        },
        "vertex_keys": {...}
      },
      "gemini-2.5-flash": {...},
      "gemini-2.0-flash": {...},
      # ... all other models in rate_limit_data
    },
    "summary": {
      "total_keys": 10,
      "total_models": 5,  # All models in the system
      "total_available": 7,
      "total_exhausted": 2,
      "total_inactive": 1,
      "displayed_models": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"]  # Optional: hint for UI
    }
  }
  ```
- **Note**: Backend returns all data; frontend handles display logic (showing 3 initially, expanding on click)
- **Advantage**: More informative than v1 - shows per-model status, usage metrics, expandable UI

---

### 2.4 Utility Methods

#### `get_usage_stats(api_key, model_name) -> Optional[Dict]`
- **v1**: ✅ Exists - queries external DB
- **v2**: ❌ Missing (but data is in DataFrame)
- **Usage**: Used internally by `get_next_working_key()` in v1
- **Migration**: Query DataFrame to return usage stats

#### `is_key_valid(key) -> bool`
- **v1**: ✅ Exists - checks `key_failure_counts[key] < MAX_FAILURES`
- **v2**: ❌ Missing (but can check `is_active` flag)
- **Usage**: Internal use in v1
- **Migration**: Check `is_active` flag in DataFrame

#### `get_fail_count(key) -> int`
- **v1**: ✅ Exists
- **v2**: ❌ Missing (no failure count tracking)
- **Usage**: Monitoring/debugging
- **Migration**: Not needed - v2 uses flags instead

#### `get_first_valid_key() -> str`
- **v1**: ✅ Exists
- **v2**: ❌ Missing
- **Usage**: Not found in codebase
- **Migration**: Use `get_key()` with any model

#### `get_random_valid_key() -> str`
- **v1**: ✅ Exists - used by `handle_api_failure()`
- **v2**: ❌ Missing
- **Usage**: Internal to `handle_api_failure()`
- **Migration**: Use `get_key()` instead

---

### 2.5 Reset Methods

#### `reset_usage()` (v2) ⚠️ **UPDATED UNDERSTANDING**
- **v2**: ✅ Exists - resets RPM/TPM/RPD
- **Behavior**: 
  - Called automatically by background task `reset_usage_bg()`
  - **ALSO** can be called manually from UI (as per user requirement)
- **Usage**: 
  - Automatic: Background task
  - Manual: Admin UI trigger (replaces v1's `reset_failure_counts()`)
- **Audit Result**: ✅ **SUFFICIENT** - handles both automatic and manual resets

#### `reset_key_failure_count(key) -> bool` (v1 only)
- **v1**: ✅ Exists - resets single key's failure count
- **Usage**: 
  - `app/router/gemini_routes.py:376, 473, 517, 571`
  - `app/scheduler/scheduled_tasks.py:74`
  - Admin UI
- **Migration**: Not needed - v2 uses `is_active`/`is_exhausted` flags instead of failure counts
- **Note**: If manual reactivation needed, can use `update_usage()` to set `is_active=True`

---

### 2.6 Instance Management ⚠️ **UPDATED DESIGN**

#### KeyManager Instance Creation
- **v1**: Singleton pattern with `get_key_manager_instance()` - complex state preservation
- **v2 Design**: **Create instance in `app.lifespan`** before app starts
  - Create in `_setup_database_and_config()` function
  - Store in `app.state.key_manager`
  - Initialize with `await key_manager.init()`
- **Usage**: 
  - `app/core/application.py:46` (lifespan startup)
  - `app/dependencies.py:21` (dependency injection)
  - `app/service/files/files_service.py:38`
  - `app/handler/retry_handler.py:21`
- **Migration**: 
  - Create instance in lifespan startup
  - Store in `app.state.key_manager`
  - Update `get_key_manager()` dependency to return `app.state.key_manager`
  - **No singleton pattern needed** - instance lives in app.state

#### `reset_key_manager_instance()` ❌ **NOT NEEDED**
- **v1**: ✅ Exists - saves state before reset
- **v2**: ❌ **Not needed** - instance created once at startup, no reset required
- **Usage**: 
  - `app/service/config/config_service.py:117, 219` (config updates)
- **Migration**: 
  - If config changes require KeyManager reinit: call `await key_manager.shutdown()` then recreate
  - State persists in database, so no state preservation needed

---

## 3. Error Handling Pattern Analysis

### Current v1 Pattern
```python
# In retry handler or chat service
try:
    response = await api_call(api_key)
except Exception as e:
    status_code = e.args[0]
    if status_code == 429:
        await set_key_exhausted_status(api_key, model, True)
    
    new_key = await key_manager.handle_api_failure(api_key, model, retries)
    if new_key:
        api_key = new_key  # Retry with new key
    else:
        raise  # No more keys
```

### Proposed v2 Pattern
```python
# Option 1: Wrapper method (backward compatible)
async def handle_api_failure(self, api_key, model_name, retries):
    # Determine error type
    error_type = "429" if status_code == 429 else "permanent"
    
    # Update usage (marks key as exhausted/inactive)
    await self.update_usage(
        model_name=model_name,
        key_value=api_key,
        is_vertex_key=api_key in self.vertex_api_keys,
        tokens_used=0,
        error=True,
        error_type=error_type
    )
    
    # Return new key for retry
    if retries < settings.MAX_RETRIES:
        try:
            return await self.get_key(model_name, is_vertex_key=False)
        except Exception:
            return ""  # No available keys
    return ""
```

### Status Code Handling
- **429 (Rate Limit)**: Set `is_exhausted=True` for that model
- **401/403 (Auth Error)**: Set `is_active=False` for all models (permanent)
- **Other Errors**: Increment failure count in v1, but v2 doesn't track counts

**Decision Needed**: Should v2 track failure counts for non-429 errors, or just use `is_active` flag?

---

## 4. Reset Functionality Audit

### v1 Reset Methods
1. `reset_failure_counts()` - Manual reset all
2. `reset_key_failure_count(key)` - Manual reset one
3. `reset_vertex_failure_counts()` - Manual reset vertex
4. `reset_vertex_key_failure_count(key)` - Manual reset one vertex

### v2 Reset Methods
1. `reset_usage()` - Automatic (minute/day resets)
2. `reset_usage_bg()` - Background task that calls `reset_usage()`

### Usage Analysis
- **Manual Resets**: Used in admin UI and scheduled tasks to recover from failures
- **Automatic Resets**: v2 handles rate limit resets automatically

### Gap Identified
- **v2 Missing**: Manual reactivation of keys after permanent errors
- **Solution**: Add `reactivate_key(key, model_name=None)` method
  - If `model_name` provided: reactivate for that model only
  - If `model_name=None`: reactivate for all models

---

## 5. Monitoring & Status Display Audit

### Current v1 Data Structure
```python
{
  "valid_keys": {"key1": 0, "key2": 1},  # key: fail_count
  "invalid_keys": {"key3": 5}
}
```

### Proposed v2 Data Structure (`get_state()`)
```python
{
  "models": {
    "gemini-pro": {
      "model_name": "gemini-pro",
      "regular_keys": {
        "available": [
          {
            "api_key": "key1",
            "rpm": 10, "tpm": 1000, "rpd": 50,
            "max_rpm": 15, "max_tpm": 1000000, "max_rpd": 1500,
            "rpm_left": 5, "tpm_left": 999000, "rpd_left": 1450,
            "last_used": "2024-01-01T12:00:00Z"
          }
        ],
        "exhausted": [...],
        "inactive": [...]
      },
      "vertex_keys": {...}
    }
  },
  "summary": {
    "total_keys": 10,
    "total_models": 5,
    "total_available": 7,
    "total_exhausted": 2,
    "total_inactive": 1
  }
}
```

### Advantages
- ✅ Shows per-model status (v1 doesn't)
- ✅ Shows actual usage metrics (v1 only shows fail counts)
- ✅ Shows available capacity (rpm_left, tpm_left, rpd_left)
- ✅ Better for table display in monitoring UI

---

## 6. Singleton Pattern Analysis

### v1 Singleton Pattern
- **Complexity**: High - preserves failure counts, cycle positions, old key lists
- **State Preservation**: Saves state before reset, restores on recreate
- **Thread Safety**: Uses `asyncio.Lock`

### v2 Singleton Pattern (Proposed)
- **Complexity**: Low - just lock and instance check
- **State Preservation**: Not needed - state is in database
- **Thread Safety**: Uses `asyncio.Lock`
- **Advantage**: Simpler, cleaner, state persists in DB anyway

### Implementation
```python
_singleton_instance: Optional[KeyManager] = None
_singleton_lock = asyncio.Lock()

async def get_key_manager_instance(api_keys, vertex_api_keys) -> KeyManager:
    global _singleton_instance
    async with _singleton_lock:
        if _singleton_instance is None:
            _singleton_instance = KeyManager(...)
            await _singleton_instance.init()
        return _singleton_instance
```

---

## 7. Migration Requirements Summary ⚠️ **UPDATED**

### Must Implement (Critical)
1. ✅ `get_paid_key() -> str` - Simple return `settings.PAID_KEY`
2. ✅ `get_next_key(is_vertex_key: bool = False) -> str` - **Unified method** for both key types
3. ✅ `get_usage_stats(api_key, model_name) -> Optional[Dict]` - Query DataFrame
4. ✅ `get_state() -> dict` - Comprehensive state for monitoring (returns ALL models; UI shows 3 initially, expandable)
5. ✅ `handle_api_failure(api_key, model_name, retries, is_vertex_key=False, status_code=None) -> str` - **Unified method** that calls `update_usage()` then `get_key()`
6. ✅ Instance creation in `app.lifespan` - Create in `_setup_database_and_config()`, store in `app.state.key_manager`
7. ✅ Update `get_key_manager()` dependency - Return `app.state.key_manager`

### Should Implement (Important)
8. ✅ `reset_usage()` - Already exists, ensure it can be called manually from UI

### Not Needed (v2 Architecture Handles)
- ❌ `get_next_vertex_key()` - Combined into `get_next_key(is_vertex_key=True)`
- ❌ `handle_vertex_api_failure()` - Combined into `handle_api_failure(is_vertex_key=True)`
- ❌ `get_key_manager_instance()` - Instance created in lifespan, stored in app.state
- ❌ `reset_key_manager_instance()` - Not needed, instance persists for app lifetime
- ❌ Failure count tracking methods (v2 uses flags)
- ❌ `get_first_valid_key()`, `get_random_valid_key()` (use `get_key()`)

---

## 8. Migration Strategy ⚠️ **UPDATED - See FIX_PLAN.md for Details**

> **Note**: Detailed implementation steps are in `FIX_PLAN.md`. This section provides high-level overview.

### Phase 1: Fix Critical Issues (See FIX_PLAN.md Phase 1-3)
1. Create custom exceptions in API clients
2. Add adapter methods to v2 (12 methods)
3. Fix minimal v2 core logic changes (~15 lines)

### Phase 2: Instance Management (See FIX_PLAN.md Phase 4-5)
1. Rename files: `key_manager.py` → `key_manager_v1.py`, `key_manager_v2.py` → `key_manager.py`
2. Update instance creation in `app.lifespan`
3. Update dependencies to use `app.state`
4. Remove `get_key_manager_instance()` function

### Phase 3: Update Call Sites (See FIX_PLAN.md Phase 6-7)
1. Update call sites to use v2 signatures
2. Update UI to use v2 format
3. Extract status_code from custom exceptions

### Phase 4: Infrastructure & Testing (See FIX_PLAN.md Phase 8)
1. Add rate limit fallback
2. Database migration script
3. Comprehensive testing

**For detailed step-by-step implementation**, refer to `FIX_PLAN.md`.

---

## 9. Risk Assessment

### Low Risk
- ✅ `get_paid_key()`, `get_next_key()`, `get_next_vertex_key()` - Simple implementations
- ✅ `get_usage_stats()` - Direct DataFrame query
- ✅ Singleton pattern - Standard pattern

### Medium Risk
- ⚠️ `handle_api_failure()` - Must match v1 behavior exactly (returns new key)
- ⚠️ `get_state()` - Must provide all data needed by monitoring UI
- ⚠️ Error handling - Must handle all status codes correctly

### High Risk
- 🔴 Migration of `get_next_working_key()` → `get_key()` - Different selection logic
- 🔴 Monitoring UI updates - Must work with new `get_state()` format
- 🔴 Retry handler integration - Must work with new error handling

---

## 10. Testing Requirements

### Unit Tests
- [ ] All new methods
- [ ] Error handling scenarios
- [ ] Edge cases (empty keys, no models, etc.)

### Integration Tests
- [ ] Retry handler with v2
- [ ] Chat services with v2
- [ ] Monitoring endpoints with v2
- [ ] Singleton pattern behavior

### Migration Tests
- [ ] Backward compatibility
- [ ] Performance comparison
- [ ] State persistence

---

## 11. Open Questions

1. **Failure Count Tracking**: Should v2 track failure counts for non-429 errors, or is `is_active` flag sufficient?
   - **Recommendation**: Use `is_active` flag only - simpler, cleaner

2. **Error Type Detection**: How to determine if error is "permanent" vs "429"?
   - **Current**: Status code 429 = rate limit, others = permanent
   - **Recommendation**: Pass status_code to `handle_api_failure()`, determine internally

3. **Model Name in `handle_api_failure()`**: Some calls don't have model_name
   - **Current**: Some services call without model_name
   - **Recommendation**: Make model_name optional, handle gracefully

4. **Vertex Key Detection**: How to know if key is vertex key in `handle_api_failure()`?
   - **Recommendation**: Check `key in self.vertex_api_keys`

---

## 12. Final Recommendations ⚠️ **UPDATED**

1. ✅ **Implement unified methods** - `get_next_key(is_vertex_key)` and `handle_api_failure(..., is_vertex_key)` - Avoids code duplication
2. ✅ **Create instance in lifespan** - Store in `app.state.key_manager` - Simpler than singleton pattern
3. ✅ **Implement `get_state()` with 3 models** - Expandable UI format for monitoring
4. ✅ **Unified error handling** - Single `handle_api_failure()` method for both key types
5. ✅ **Manual reset support** - `reset_usage()` can be called from UI
6. ✅ **Test thoroughly before migration** - Especially retry handlers and instance lifecycle
7. ✅ **Migrate incrementally** - Phase by phase with testing

---

## 13. Design Decisions Summary

### ✅ Unified API Design
- `get_next_key(is_vertex_key=False)` - One method for both key types
- `handle_api_failure(..., is_vertex_key=False)` - One method for both key types
- Consistent with existing `get_key(is_vertex_key=False)` pattern

### ✅ Instance Management
- Create in `app.lifespan` startup
- Store in `app.state.key_manager`
- No singleton pattern needed
- State persists in database

### ✅ Monitoring UI
- Backend returns ALL models in `get_state()`
- Frontend displays 3 models initially: `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.0-flash`
- Expandable on click (like `keyUsageDetailsModal`) to show all models
- Rich data per model

---

## Conclusion

The v2 architecture is **superior** to v1 in every way:
- Better state management (DataFrame)
- Automatic resets (background task)
- Model-aware tracking
- Cleaner code (unified methods)
- Simpler instance management (app.state vs singleton)

**All design decisions clarified**:
- ✅ Unified methods (no duplication)
- ✅ Instance in lifespan (no singleton)
- ✅ Manual reset support
- ✅ Expandable monitoring UI

**Status**: ✅ **Ready for Implementation** - All requirements identified, design decisions finalized, no ambiguities remaining.

