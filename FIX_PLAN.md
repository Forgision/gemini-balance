# KeyManager v1 → v2 Migration Fix Plan

> **Status**: ✅ **PRIMARY IMPLEMENTATION GUIDE**  
> **Last Updated**: Based on deep assessment of 30 critical issues  
> **Related Documents**: See `MIGRATION_AUDIT.md` for architecture comparison, `IMPLEMENTATION_PLAN.md` is superseded by this document

## Strategy: Adapter Pattern Approach

**Principle**: Keep v2's core logic unchanged. Create adapter/wrapper methods that bridge v1 API to v2 implementation.

## Document Purpose

This is the **primary implementation guide** for migrating from KeyManager v1 to v2. It contains:
- Detailed fix implementations for 30 critical issues identified in deep assessment
- Step-by-step code changes with minimal v2 core logic modifications
- External changes required (API clients, call sites, UI, instance management)

**For architecture comparison and design decisions**, see `MIGRATION_AUDIT.md`.  
**For earlier implementation notes**, see `IMPLEMENTATION_PLAN.md` (superseded by this document).

---

## Phase 1: Critical Method Signatures & Adapters (HIGHEST PRIORITY)

### 1.1 Fix `handle_api_failure()` - Custom Exceptions in API Clients (NO v2 LOGIC CHANGE)

**Problem**: Call sites don't pass `status_code`, but v2 needs it.

**Solution**: Create custom exceptions in `GeminiApiClient` and `OpenaiApiClient` that hold `status_code`. When `response.status_code != 200`, raise these exceptions. Then extract `status_code` from exception and pass to `handle_api_failure()`.

**Implementation** (Modify API clients, NO v2 change):

**Step 1: Create custom exception** (New file or add to existing exceptions module):
```python
# app/exception/api_exceptions.py
class ApiClientException(Exception):
    """Base exception for API client errors."""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"API Error {status_code}: {message}")
```

**Step 2: Update GeminiApiClient** (Modify `app/service/client/api_client.py`):
```python
from app.exception.api_exceptions import ApiClientException

# In generate_content(), replace line 109:
# raise Exception(response.status_code, error_content)
# With:
raise ApiClientException(response.status_code, error_content)

# In stream_generate_content(), replace line 141:
# raise Exception(response.status_code, error_msg)
# With:
raise ApiClientException(response.status_code, error_msg)

# In count_tokens(), replace line 165:
# raise Exception(response.status_code, error_content)
# With:
raise ApiClientException(response.status_code, error_content)

# In embed_content(), replace line 192:
# raise Exception(response.status_code, error_content)
# With:
raise ApiClientException(response.status_code, error_content)

# In batch_embed_contents(), replace line 219:
# raise Exception(response.status_code, error_content)
# With:
raise ApiClientException(response.status_code, error_content)
```

**Step 3: Update OpenaiApiClient** (Modify `app/service/client/api_client.py`):
```python
# Similar changes for all methods that raise Exception with status_code
# Replace all: raise Exception(response.status_code, error_content)
# With: raise ApiClientException(response.status_code, error_content)
```

**Step 4: Update call sites to extract status_code** (Modify retry handlers and chat services):
```python
# In retry_handler.py and chat services:
try:
    # API call
except ApiClientException as e:
    status_code = e.status_code
    new_key = await key_manager.handle_api_failure(
        old_key, model_name, retries, status_code=status_code
    )
except Exception as e:
    # Fallback for other exceptions
    status_code = getattr(e, 'status_code', None) or 500
    new_key = await key_manager.handle_api_failure(
        old_key, model_name, retries, status_code=status_code
    )
```

**Step 5: Add handle_api_failure to v2** (Add to v2, NO core logic change):
```python
async def handle_api_failure(
    self, 
    api_key: str, 
    model_name: str, 
    retries: int,
    is_vertex_key: Optional[bool] = None,
    status_code: Optional[int] = None
) -> str:
    """
    Handle API call failure. Compatible with v1 signature.
    If status_code not provided, defaults to 'permanent' error.
    """
    # Auto-detect vertex key if not provided
    if is_vertex_key is None:
        is_vertex_key = api_key in self.vertex_api_keys
    
    # Determine error type (default to permanent if status_code not provided)
    if status_code == 429:
        error_type = "429"
    else:
        error_type = "permanent"
    
    # Update usage (marks key as exhausted/inactive)
    await self.update_usage(
        model_name=model_name,
        key_value=api_key,
        is_vertex_key=is_vertex_key,
        tokens_used=0,
        error=True,
        error_type=error_type
    )
    
    # Return new key for retry (match v1 behavior)
    if retries < settings.MAX_RETRIES:
        # get_key() now returns empty string instead of raising (see 1.3)
        return await self.get_key(model_name or "gemini-pro", is_vertex_key=is_vertex_key)
    return ""
```

**Changes Required**:
- ✅ Create custom exception class (new file)
- ✅ Update GeminiApiClient to raise custom exception (modify api_client.py)
- ✅ Update OpenaiApiClient to raise custom exception (modify api_client.py)
- ✅ Update call sites to catch custom exception and extract status_code
- ✅ Add method to v2 (new method, no core logic change)

---

### 1.2 Fix `get_next_key()` - Unified Method (NO v2 LOGIC CHANGE)

**Problem**: Need unified method for both key types.

**Solution**: Simple wrapper using existing cycles.

**Implementation** (Add to v2, NO core logic change):
```python
async def get_next_key(self, is_vertex_key: bool = False) -> str:
    """Get the next API key in round-robin fashion."""
    try:
        if is_vertex_key:
            return next(self.vertex_api_keys_cycle)
        else:
            return next(self.api_keys_cycle)
    except StopIteration:
        logger.warning("API key cycle is empty.")
        return ""
```

**Changes Required**:
- ✅ Add method to v2 (uses existing cycles, no core logic change)

---

### 1.3 Fix `get_key()` Exception Handling - Option B (MINIMAL v2 CHANGE)

**Problem**: `get_key()` raises exception, but retry handlers expect empty string.

**Solution**: Modify `get_key()` to return empty string instead of raising exception.

**Implementation** (MINIMAL v2 change):
```python
# In get_key(), change line 626 from:
# raise Exception(f"No available keys for model {model_name}.")
# To:
logger.warning(f"No available keys for model {model_name}, falling back to cycle.")
return await self.get_next_key(is_vertex_key=is_vertex_key)
```

**Changes Required**:
- ✅ Modify `get_key()` line 626 (2 lines change - minimal impact)

---

### 1.4 Fix `get_next_working_key()` - Update Call Sites (NO v2 LOGIC CHANGE)

**Problem**: `get_next_working_key(model_name)` needs to be replaced with `get_key()`.

**Solution**: Update all call sites to use `get_key(model_name, is_vertex_key=False)` directly.

**Implementation** (Update call sites, NO v2 change):
- Find all usages of `get_next_working_key(model_name)`
- Replace with `get_key(model_name, is_vertex_key=False)`

**Changes Required**:
- ✅ Update call sites to use v2 signature (external changes only)
- ✅ No changes to v2 class needed

---

### 1.5 Fix `get_next_working_vertex_key()` - Update Call Sites (NO v2 LOGIC CHANGE)

**Problem**: v1 method doesn't take model_name, but v2's `get_key()` requires it.

**Solution**: Update all call sites to use `get_key(model_name, is_vertex_key=True)` with appropriate model_name.

**Implementation** (Update call sites, NO v2 change):
- Find all usages of `get_next_working_vertex_key()`
- Determine model_name from context (request, default, etc.)
- Replace with `get_key(model_name, is_vertex_key=True)`

**Changes Required**:
- ✅ Update call sites to use v2 signature (external changes only)
- ✅ Update dependencies that extract model_name from request
- ✅ No changes to v2 class needed

---

### 1.6 Fix `get_random_valid_key()` - Use Cycles (NO v2 LOGIC CHANGE)

**Problem**: Still used in 5 locations, v2 doesn't have it.

**Solution**: Create adapter that returns from cycles (simpler approach).

**Implementation** (Add to v2, NO core logic change):
```python
async def get_random_valid_key(self, model_name: Optional[str] = None) -> str:
    """
    Adapter method for v1 compatibility.
    Returns a key from cycle (round-robin).
    """
    return await self.get_next_key(is_vertex_key=False)
```

**Changes Required**:
- ✅ Add method to v2 (simple wrapper using cycles, no core logic change)

---

## Phase 2: Status & Monitoring Adapters (HIGH PRIORITY)

### 2.1 Fix `get_keys_by_status()` - Adapt v2 Format & Update UI (NO v2 LOGIC CHANGE)

**Problem**: Call sites expect v1 format, but we should use v2 format and update UI.

**Solution**: Create method that returns v2 format, then update UI to use new format.

**Implementation** (Add to v2, NO core logic change):
```python
async def get_keys_by_status(self) -> dict:
    """
    Returns keys grouped by status in v2 format.
    Uses is_active and is_exhausted flags from DataFrame.
    """
    if not self.is_ready:
        await self._check_ready()
    
    valid_keys = {}
    invalid_keys = {}
    
    async with self.lock:
        # Get all unique keys from DataFrame
        all_keys = set(self.df.index.get_level_values("api_key"))
        
        for key in all_keys:
            # Check if key is active for any model
            key_rows = self.df[self.df.index.get_level_values("api_key") == key]
            is_any_active = key_rows["is_active"].any() if not key_rows.empty else False
            is_any_exhausted = key_rows["is_exhausted"].any() if not key_rows.empty else False
            
            # v2 format: use status flags
            if is_any_active and not is_any_exhausted:
                valid_keys[key] = {"status": "active", "exhausted": False}
            elif is_any_exhausted:
                valid_keys[key] = {"status": "exhausted", "exhausted": True}
            else:
                invalid_keys[key] = {"status": "inactive", "exhausted": False}
    
    return {"valid_keys": valid_keys, "invalid_keys": invalid_keys}
```

**Changes Required**:
- ✅ Add method to v2 (returns v2 format, no core logic change)
- ✅ Update UI to use new format (external changes)

---

### 2.2 Fix `get_all_keys_with_fail_count()` - Adapter Method (NO v2 LOGIC CHANGE)

**Problem**: Call sites expect v1 format with fail_count.

**Solution**: Create adapter that returns v1 format.

**Implementation** (Add to v2, NO core logic change):
```python
async def get_all_keys_with_fail_count(self) -> dict:
    """
    Adapter method for v1 compatibility.
    Returns all keys with status. Since v2 doesn't track failure counts,
    uses is_active flag (0 = valid, 1 = invalid).
    """
    status = await self.get_keys_by_status()
    all_keys = {**status["valid_keys"], **status["invalid_keys"]}
    
    return {
        "valid_keys": status["valid_keys"],
        "invalid_keys": status["invalid_keys"],
        "all_keys": all_keys
    }
```

**Changes Required**:
- ✅ Add method to v2 (adapter, no core logic change)

---

### 2.3 Implement `get_state()` - New Method & Update UI (NO v2 LOGIC CHANGE)

**Problem**: Need comprehensive state for monitoring.

**Solution**: Query existing DataFrame, format for UI. Update UI to use new format.

**Implementation** (Add to v2, NO core logic change):
```python
async def get_state(self) -> dict:
    """
    Get comprehensive state of all keys for monitoring.
    Returns all models (UI handles display of 3 initially).
    """
    if not self.is_ready:
        await self._check_ready()
    
    async with self.lock:
        df_copy = self.df.copy()
    
    state = {
        "models": {},
        "summary": {
            "total_keys": len(self.api_keys) + len(self.vertex_api_keys),
            "total_models": len(self.rate_limit_models),
            "total_available": 0,
            "total_exhausted": 0,
            "total_inactive": 0,
        }
    }
    
    for model in self.rate_limit_models:
        if model not in df_copy.index.get_level_values("model_name"):
            continue
        
        model_data = {
            "model_name": model,
            "regular_keys": {"available": [], "exhausted": [], "inactive": []},
            "vertex_keys": {"available": [], "exhausted": [], "inactive": []}
        }
        
        # Process regular keys
        try:
            regular_df = df_copy.xs((model, False), level=["model_name", "is_vertex_key"])
            for api_key, row in regular_df.iterrows():
                key_info = self._format_key_info(api_key, row)
                if not row["is_active"]:
                    model_data["regular_keys"]["inactive"].append(key_info)
                    state["summary"]["total_inactive"] += 1
                elif row["is_exhausted"]:
                    model_data["regular_keys"]["exhausted"].append(key_info)
                    state["summary"]["total_exhausted"] += 1
                else:
                    model_data["regular_keys"]["available"].append(key_info)
                    state["summary"]["total_available"] += 1
        except KeyError:
            pass
        
        # Process vertex keys (similar logic)
        # ... (same pattern for vertex keys)
        
        state["models"][model] = model_data
    
    return state

def _format_key_info(self, api_key, row) -> dict:
    """Helper to format key information."""
    return {
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
```

**Changes Required**:
- ✅ Add method to v2 (new method, queries existing DataFrame, no core logic change)
- ✅ Update UI to use new format (external changes)

---

## Phase 3: Utility Methods (MEDIUM PRIORITY)

### 3.1 Implement `get_usage_stats()` - Query DataFrame (NO v2 LOGIC CHANGE)

**Problem**: Need to return usage stats for a key/model.

**Solution**: Query existing DataFrame.

**Implementation** (Add to v2, NO core logic change):
```python
async def get_usage_stats(
    self, api_key: str, model_name: str
) -> Optional[Dict[str, Any]]:
    """
    Get usage statistics for a given key and model.
    Adapter method for v1 compatibility.
    """
    if not self.is_ready:
        await self._check_ready()
    
    match, model_name = self._model_normalization(model_name)
    is_vertex_key = api_key in self.vertex_api_keys
    
    async with self.lock:
        try:
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

**Changes Required**:
- ✅ Add method to v2 (queries existing DataFrame, no core logic change)

---

### 3.2 Implement `get_paid_key()` - Simple Return (NO v2 LOGIC CHANGE)

**Problem**: Missing method.

**Solution**: Simple return from settings.

**Implementation** (Add to v2, NO core logic change):
```python
async def get_paid_key(self) -> str:
    """Get the paid API key for premium features."""
    from app.config.config import settings
    return settings.PAID_KEY
```

**Changes Required**:
- ✅ Add method to v2 (simple return, no core logic change)

---

### 3.3 Create `reset_key_failure_count()` - Manual Reactivation (NO v2 LOGIC CHANGE)

**Problem**: `reset_key_failure_count()` used in scheduled tasks and admin UI for manual reactivation.

**Note**: v2 has auto-reset functionality, but manual reactivation is still needed for immediate key reactivation (before auto-reset).

**Solution**: Create adapter that reactivates key by setting flags.

**Implementation** (Add to v2, NO core logic change):
```python
async def reset_key_failure_count(self, key: str) -> bool:
    """
    Manually reactivate a key (adapter for v1 compatibility).
    Reactivates for all models.
    Note: Auto-reset will handle periodic resets, but this allows immediate reactivation.
    """
    if not self.is_ready:
        await self._check_ready()
    
    is_vertex_key = key in self.vertex_api_keys
    
    async with self.lock:
        try:
            # Reactivate for all models
            key_mask = self.df.index.get_level_values("api_key") == key
            if key_mask.any():
                self.df.loc[key_mask, "is_active"] = True
                self.df.loc[key_mask, "is_exhausted"] = False
                await self._on_update_usage()
                logger.info(f"Manually reactivated key: {redact_key_for_logging(key)}")
                return True
            else:
                logger.warning(f"Key not found for reactivation: {redact_key_for_logging(key)}")
                return False
        except Exception as e:
            logger.error(f"Error reactivating key: {e}")
            return False
```

**Changes Required**:
- ✅ Add method to v2 (manual reactivation, no core logic change)

---

## Phase 4: Instance Management (HIGH PRIORITY)

### 4.1 Remove `get_key_manager_instance()` - Use app.state Only (NO v2 LOGIC CHANGE)

**Problem**: `get_key_manager_instance()` is not needed in future. Instance should come from app.state.

**Solution**: Remove `get_key_manager_instance()` wrapper. Update all call sites to use app.state directly.

**Implementation** (External changes only, NO v2 change):

**Step 1: Update `app/dependencies.py`**:
```python
from fastapi import Request

async def get_key_manager(request: Request) -> KeyManager:
    """Get KeyManager instance from app.state."""
    if not hasattr(request.app.state, "key_manager"):
        raise RuntimeError("KeyManager not initialized. Check application startup.")
    return request.app.state.key_manager
```

**Step 2: Update all call sites**:
- Replace `await get_key_manager_instance(...)` with dependency injection or `request.app.state.key_manager`
- For non-request contexts (scheduled tasks, etc.), pass app instance or use alternative approach

**Changes Required**:
- ✅ Update `app/dependencies.py` to use app.state
- ✅ Update all call sites to use dependency injection or app.state
- ✅ Remove `get_key_manager_instance()` function (not needed)

---

### 4.2 Rename Files & Update Application Lifespan (NO v2 LOGIC CHANGE)

**Problem**: Need to rename files and create instance in lifespan.

**Solution**: 
1. Rename `key_manager.py` → `key_manager_v1.py`
2. Rename `key_manager_v2.py` → `key_manager.py`
3. Update lifespan to create instance and store in app.state

**Implementation**:

**Step 1: Rename files**:
```bash
# Rename key_manager.py to key_manager_v1.py
# Rename key_manager_v2.py to key_manager.py
```

**Step 2: Update imports** (Update all files that import from key_manager):
- Change `from app.service.key.key_manager import ...` to use new file names
- For v1 imports: `from app.service.key.key_manager_v1 import ...`
- For v2 imports: `from app.service.key.key_manager import ...` (now points to v2)

**Step 3: Update lifespan** (Modify `app/core/application.py`, NO v2 change):
```python
async def _setup_database_and_config(app: FastAPI, app_settings):
    """Initializes database, syncs settings, and initializes KeyManager."""
    initialize_database()
    logger.info("Database initialized successfully")
    await connect_to_db()
    await sync_initial_settings()
    
    # Create KeyManager instance
    from app.service.key.key_manager_v2 import KeyManager, AsyncSessionLocal
    key_manager = KeyManager(
        api_keys=app_settings.API_KEYS,
        vertex_api_keys=app_settings.VERTEX_API_KEYS,
        async_session_maker=AsyncSessionLocal,
    )
    
    # Initialize
    success = await key_manager.init()
    if not success:
        raise RuntimeError("Failed to initialize KeyManager")
    
    # Store in app.state
    app.state.key_manager = key_manager
    logger.info("Database, config sync, and KeyManager initialized successfully")

async def _shutdown_database(app: FastAPI):
    """Disconnects from the database and shuts down KeyManager."""
    if hasattr(app.state, "key_manager") and app.state.key_manager:
        await app.state.key_manager.shutdown()
    await disconnect_from_db()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ... existing code ...
    await _setup_database_and_config(app, settings)
    # ... existing code ...
    yield
    # ... existing code ...
    await _shutdown_database(app)
```

**Changes Required**:
- ✅ Update `app/core/application.py` (no v2 class change)

---

### 4.3 Update Dependencies (NO v2 LOGIC CHANGE)

**Problem**: `get_key_manager()` dependency needs to return app.state.key_manager.

**Solution**: Update dependency function.

**Implementation** (Modify `app/dependencies.py`, NO v2 change):
```python
async def get_key_manager(request: Request) -> KeyManager:
    """Get KeyManager instance from app.state."""
    if not hasattr(request.app.state, "key_manager"):
        raise RuntimeError("KeyManager not initialized. Check application startup.")
    return request.app.state.key_manager
```

**Changes Required**:
- ✅ Update `app/dependencies.py` (no v2 class change)

---

### 4.4 Fix Config Service Reinitialization - Clarify Instance Creation (NO v2 LOGIC CHANGE)

**Problem**: Config service needs to reinitialize KeyManager when config changes. Need to clarify how to access app instance.

**Solution**: Pass app instance to config service methods, or use a different approach.

**Option A: Pass app instance** (Recommended):
```python
# In update_config method signature:
async def update_config(config_data: Dict[str, Any], app: FastAPI) -> Dict[str, Any]:
    # ... existing config update code ...
    
    # Reinitialize KeyManager
    if hasattr(app.state, "key_manager") and app.state.key_manager:
        await app.state.key_manager.shutdown()
    
    from app.service.key.key_manager import KeyManager, AsyncSessionLocal
    key_manager = KeyManager(
        api_keys=settings.API_KEYS,
        vertex_api_keys=settings.VERTEX_API_KEYS,
        async_session_maker=AsyncSessionLocal,
    )
    success = await key_manager.init()
    if not success:
        raise RuntimeError("Failed to reinitialize KeyManager")
    app.state.key_manager = key_manager
```

**Option B: Store app reference** (Alternative):
```python
# Store app reference in ConfigService
class ConfigService:
    _app: Optional[FastAPI] = None
    
    @classmethod
    def set_app(cls, app: FastAPI):
        cls._app = app
    
    @staticmethod
    async def update_config(config_data: Dict[str, Any]) -> Dict[str, Any]:
        # ... existing code ...
        if ConfigService._app:
            # Reinitialize KeyManager using stored app reference
            # ... same as Option A ...
```

**Option C: Use dependency injection in routes** (Best practice):
```python
# In routes, pass app via dependency
@router.post("/config")
async def update_config(
    config_data: Dict[str, Any],
    app: FastAPI = Depends(get_app)  # Need to create get_app dependency
):
    return await ConfigService.update_config(config_data, app)
```

**Changes Required**:
- ✅ Update `app/service/config/config_service.py` to accept app parameter
- ✅ Update route handlers to pass app instance
- ✅ Update `reset_config()` similarly

---

### 4.5 Fix Files Service Lazy Init (NO v2 LOGIC CHANGE)

**Problem**: Files service has lazy init pattern.

**Solution**: Update to use app.state or dependency injection.

**Implementation** (Modify `app/service/files/files_service.py`, NO v2 change):
```python
# Option 1: Use dependency injection
async def _get_key_manager(self, request: Request) -> KeyManager:
    """Get KeyManager from app.state."""
    return request.app.state.key_manager

# Option 2: Remove lazy init, use dependency injection in routes
```

**Changes Required**:
- ✅ Update `app/service/files/files_service.py` (no v2 class change)

---

## Phase 5: Error Handling Fixes (MEDIUM PRIORITY)

### 5.1 Fix Status Code Extraction in Call Sites (NO v2 LOGIC CHANGE)

**Problem**: Call sites need to extract status_code before calling `handle_api_failure()`.

**Solution**: Update call sites to extract status_code.

**Implementation** (Modify call sites, NO v2 change):

**In retry_handler.py**:
```python
# Extract status_code from exception
status_code = None
if hasattr(e, 'args') and len(e.args) > 0:
    status_code = e.args[0] if isinstance(e.args[0], int) else None

new_key = await key_manager.handle_api_failure(
    old_key, model_name, retries, status_code=status_code
)
```

**In chat services** (already have status_code):
```python
# They already extract status_code, just pass it
new_api_key = await self.key_manager.handle_api_failure(
    current_attempt_key, model, retries, status_code=status_code
)
```

**Changes Required**:
- ✅ Update call sites (no v2 class change)

---

### 5.2 Fix Unknown Model Handling in `update_usage()` - Minimal Change

**Problem**: Unknown models silently fail.

**Solution**: Create entry on-the-fly with unlimited limits.

**Implementation** (MINIMAL v2 change - add to update_usage):
```python
# In update_usage(), after line 673, add:
if model_name not in self.df.index.get_level_values("model_name"):
    # Create entry for unknown model with unlimited limits
    await self._create_unknown_model_entry(model_name, key_value, is_vertex_key)
    # Continue with update
```

**Add helper method** (NO core logic change):
```python
async def _create_unknown_model_entry(
    self, model_name: str, api_key: str, is_vertex_key: bool
):
    """Create DataFrame entry for unknown model with unlimited limits."""
    async with self.lock:
        new_entry = {
            "api_key": api_key,
            "model_name": model_name,
            "rpm": 0,
            "max_rpm": 999999,  # Unlimited
            "tpm": 0,
            "max_tpm": 999999999,  # Unlimited
            "rpd": 0,
            "max_rpd": 999999,  # Unlimited
            "minute_reset_time": self.now_minute(),
            "day_reset_time": self.now_day(),
            "last_used": self.now() - timedelta(2),
            "is_vertex_key": is_vertex_key,
            "is_active": True,
            "is_exhausted": False,
        }
        new_df = pd.DataFrame([new_entry])
        new_df.set_index(self._INDEX_LEVEL, inplace=True, drop=True)
        self.df = pd.concat([self.df, new_df])
```

**Changes Required**:
- ✅ Add helper method to v2 (new method, no core logic change)
- ✅ Add 3 lines to update_usage() (minimal change)

---

## Phase 6: Bug Fixes (LOW PRIORITY - But Easy)

### 6.1 Fix Duplicate Assignment Bug

**Problem**: Line 407 duplicates line 406.

**Solution**: Remove duplicate line.

**Implementation** (MINIMAL v2 change):
```python
# Remove line 407:
# self.last_day_reset_ts = max_day_reset_dt
```

**Changes Required**:
- ✅ Remove one line (minimal change)

---

### 6.2 Fix `update_usage()` Return Value

**Problem**: Line 686 returns without value.

**Solution**: Make return explicit.

**Implementation** (MINIMAL v2 change):
```python
# Change line 686 from:
# return
# To:
return True  # Error was handled
```

**Changes Required**:
- ✅ Change one line (minimal change)

---

### 6.3 Fix `get_key()` Exception vs Fallback

**Problem**: Inconsistent - falls back for unknown model, raises for no keys.

**Solution**: Make consistent - always fallback.

**Implementation** (MINIMAL v2 change):
```python
# Change line 626 from:
# raise Exception(f"No available keys for model {model_name}.")
# To:
logger.warning(f"No available keys for model {model_name}, falling back to cycle.")
return await self.get_next_key(is_vertex_key=is_vertex_key)
```

**Changes Required**:
- ✅ Change 2 lines (minimal change, improves consistency)

---

## Phase 7: Infrastructure Fixes (MEDIUM PRIORITY)

### 7.1 Add Rate Limit Scraping Fallback (NO v2 LOGIC CHANGE)

**Problem**: Scraping failure causes init to fail.

**Solution**: Add fallback to cached/default data.

**Implementation** (MINIMAL v2 change - add to init):
```python
# In init(), after line 512, add fallback:
try:
    rate_limit_data = scrape_gemini_rate_limits(GEMINI_RATE_LIMIT_URL)
    if not rate_limit_data:
        raise ValueError("rate_limit_data is empty!")
except Exception as e:
    logger.warning(f"Failed to scrape rate limits: {e}. Using cached/default.")
    # Use cached or default rate limits
    # Could load from file or use hardcoded defaults
    rate_limit_data = self._get_default_rate_limits()

# Add helper method
def _get_default_rate_limits(self) -> dict:
    """Return default rate limits if scraping fails."""
    # Return hardcoded defaults or load from cache file
    return {"Free Tier": {...}}  # Default structure
```

**Changes Required**:
- ✅ Add try-except in init() (minimal change)
- ✅ Add helper method (new method, no core logic change)

---

### 7.2 Handle "Free Tier" Key Missing (NO v2 LOGIC CHANGE)

**Problem**: Hardcoded "Free Tier" key might not exist.

**Solution**: Add fallback.

**Implementation** (MINIMAL v2 change):
```python
# In init(), after line 516, add:
if "Free Tier" not in rate_limit_data:
    logger.warning("Free Tier not found, using first available tier.")
    tier_key = list(rate_limit_data.keys())[0] if rate_limit_data else "Free Tier"
    self.rate_limit_data = rate_limit_data.get(tier_key, {}).copy()
else:
    self.rate_limit_data = rate_limit_data["Free Tier"].copy()
```

**Changes Required**:
- ✅ Add 3 lines (minimal change)

---

### 7.3 Database Migration Script (NO v2 LOGIC CHANGE)

**Problem**: Need to migrate v1 data to v2 database.

**Solution**: Create migration script (separate file).

**Implementation** (New file, NO v2 change):
```python
# migration_script.py
async def migrate_v1_to_v2():
    """Migrate data from v1 database to v2 database."""
    # 1. Read from v1 database (get_usage_stats_by_key_and_model)
    # 2. Transform to v2 format
    # 3. Write to v2 database (UsageMatrix)
    pass
```

**Changes Required**:
- ✅ Create migration script (separate file, no v2 change)

---

## Phase 8: Dependency Updates (HIGH PRIORITY)

### 8.1 Update `dep_get_next_working_vertex_key()` (NO v2 LOGIC CHANGE)

**Problem**: Dependency needs model_name.

**Solution**: Extract from request or use default.

**Implementation** (Modify `app/router/vertex_express_routes.py`, NO v2 change):
```python
async def dep_get_next_working_vertex_key(
    request: Request,
    key_manager: KeyManager = Depends(get_key_manager)
) -> str:
    """Get the next available Vertex API key."""
    # Try to extract model from request if available
    # Otherwise use default
    model_name = getattr(request.state, "model_name", None) or "gemini-pro"
    return await key_manager.get_next_working_vertex_key(model_name=model_name)
```

**Changes Required**:
- ✅ Update dependency (no v2 class change)

---

### 8.2 Update `get_next_working_key_wrapper()` (NO v2 LOGIC CHANGE)

**Problem**: Hardcoded model_name.

**Solution**: Extract from request.

**Implementation** (Modify `app/router/openai_routes.py`, NO v2 change):
```python
async def get_next_working_key_wrapper(
    request: Request,
    key_manager: KeyManager = Depends(get_key_manager)
) -> str:
    """Get the next available API key, extracting model from request if possible."""
    # Try to get model from request body/query params
    # Fallback to default
    model_name = "gemini-pro"  # Default
    # Could parse request body if needed
    return await key_manager.get_next_working_key(model_name=model_name)
```

**Changes Required**:
- ✅ Update wrapper (no v2 class change)

---

## Summary: Changes to v2 Class

### ✅ Methods to ADD (No Core Logic Changes):
1. `get_paid_key()` - Simple return
2. `get_next_key(is_vertex_key)` - Uses existing cycles
3. `get_random_valid_key()` - Uses cycles (simple wrapper)
4. `get_usage_stats()` - Queries existing DataFrame
5. `get_keys_by_status()` - Returns v2 format, queries DataFrame
6. `get_all_keys_with_fail_count()` - Adapter, uses get_keys_by_status()
7. `get_state()` - Queries existing DataFrame
8. `handle_api_failure()` - Wrapper, uses update_usage() + get_key()
9. `reset_key_failure_count()` - Manual reactivation
10. `_create_unknown_model_entry()` - Helper method
11. `_format_key_info()` - Helper method
12. `_get_default_rate_limits()` - Helper method

### ⚠️ MINIMAL Changes to Existing Methods:
1. `get_key()` line 626 - Change exception to fallback (2 lines)
2. `update_usage()` line 686 - Add explicit return (1 line)
3. `update_usage()` - Add unknown model handling (3 lines)
4. `_load_from_db()` line 407 - Remove duplicate (1 line)
5. `init()` - Add rate limit fallback (5 lines)
6. `init()` - Handle missing "Free Tier" (3 lines)

### ✅ External Changes (No v2 Logic Changes):
1. Create custom exception class (`ApiClientException`)
2. Update `GeminiApiClient` to raise custom exception
3. Update `OpenaiApiClient` to raise custom exception
4. Update call sites to extract status_code from exception
5. Update call sites to use `get_key()` instead of `get_next_working_key()`
6. Update call sites to use `get_key()` instead of `get_next_working_vertex_key()`
7. Update UI to use v2 format for `get_keys_by_status()`
8. Update UI to use `get_state()` format
9. Rename files: `key_manager.py` → `key_manager_v1.py`, `key_manager_v2.py` → `key_manager.py`
10. Update all imports after file rename
11. Update `app/core/application.py` to create instance in lifespan
12. Update `app/dependencies.py` to use app.state
13. Update `app/service/config/config_service.py` to reinitialize via app.state
14. Remove `get_key_manager_instance()` function

### ✅ NO Changes to Core Logic:
- DataFrame structure unchanged
- Reset logic unchanged
- Background task unchanged
- Lock usage unchanged
- Database operations unchanged

---

## Implementation Order

1. **Phase 1**: Create custom exceptions and update API clients (1.1) - External changes
2. **Phase 2**: Minimal v2 changes (1.3, 6.1-6.3) - Fix get_key(), bugs
3. **Phase 3**: Add adapter methods to v2 (1.2, 1.6, 2.1, 2.3, 3.1-3.3) - 8 methods
4. **Phase 4**: Rename files and update imports (4.2) - File operations
5. **Phase 5**: Update instance management (4.1, 4.2, 4.3, 4.4, 4.5) - External changes
6. **Phase 6**: Update call sites (1.4, 1.5, 5.1) - External changes
7. **Phase 7**: Update UI (2.1, 2.3) - External changes
8. **Phase 8**: Infrastructure fixes (5.2, 7.1-7.3) - Minimal v2 + external changes

---

## Risk Assessment After Fixes

- **Low Risk**: Adapter methods (just wrappers)
- **Low Risk**: External changes (app/core, dependencies)
- **Very Low Risk**: Minimal v2 changes (1-5 lines each)
- **Medium Risk**: Database migration (separate script)

**Total v2 Core Logic Changes**: ~15 lines across 4 methods
**Total New Methods**: 15 adapter/wrapper methods
**Total External Changes**: ~10 files (app/core, dependencies, routes, services)

---

## Final Recommendation

✅ **SAFE TO PROCEED** after implementing this plan.

**Key Principle**: Keep v2's core logic intact. All compatibility achieved through:
- Adapter/wrapper methods (15 new methods)
- Minimal changes to existing methods (~15 lines)
- External updates (dependencies, lifespan, routes)

**Estimated Implementation Time**: 1-2 days

