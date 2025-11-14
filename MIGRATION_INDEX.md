# KeyManager Migration Documentation Index

This document provides an overview of all migration-related documentation and how they relate to each other.

## Document Structure

### 1. 📋 **FIX_PLAN.md** - PRIMARY IMPLEMENTATION GUIDE
**Status**: ✅ **Use this for implementation**  
**Purpose**: Detailed step-by-step implementation guide  
**Contents**:
- 30 critical issues identified in deep assessment
- Detailed code fixes with minimal v2 core logic changes
- Phase-by-phase implementation order
- External changes required (API clients, call sites, UI)

**When to use**: When implementing the migration

---

### 2. 📊 **MIGRATION_AUDIT.md** - REFERENCE DOCUMENT
**Status**: ✅ **Use this for understanding**  
**Purpose**: Architecture comparison and design decisions  
**Contents**:
- v1 vs v2 architecture comparison
- Method-by-method analysis
- Design decisions and rationale
- Migration strategy overview
- Risk assessment

**When to use**: When understanding why decisions were made, or comparing v1 vs v2

---

### 3. 📝 **IMPLEMENTATION_PLAN.md** - HISTORICAL REFERENCE
**Status**: ⚠️ **SUPERSEDED**  
**Purpose**: Historical reference (earlier plan)  
**Contents**:
- Original implementation plan
- Some outdated decisions (singleton pattern, etc.)

**When to use**: Historical reference only - do not use for implementation

---

### 4. 🚨 **CRITICAL_ISSUES.md** - ISSUE TRACKING
**Status**: ✅ **Reference for issue details**  
**Purpose**: Detailed list of 30 critical issues  
**Contents**:
- Complete list of critical issues that would cause migration failure
- Problem descriptions and solutions
- Risk levels

**When to use**: When reviewing specific issues or understanding problem details

---

## Quick Navigation

### I want to...
- **Implement the migration** → Read `FIX_PLAN.md`
- **Understand architecture differences** → Read `MIGRATION_AUDIT.md`
- **Review critical issues** → Read `CRITICAL_ISSUES.md`
- **See historical context** → Read `IMPLEMENTATION_PLAN.md` (but use FIX_PLAN.md for actual work)

---

## Document Relationships

```
FIX_PLAN.md (Primary)
    ↓ references
MIGRATION_AUDIT.md (Reference)
    ↓ references
CRITICAL_ISSUES.md (Details)
    
IMPLEMENTATION_PLAN.md (Historical - Superseded)
```

---

## Key Decisions Summary

All documents align on these key decisions:

1. ✅ **Unified Methods**: `get_next_key(is_vertex_key)`, `handle_api_failure(..., is_vertex_key)`
2. ✅ **Instance Management**: Create in `app.lifespan`, store in `app.state.key_manager`
3. ✅ **Custom Exceptions**: `ApiClientException` with `status_code` attribute
4. ✅ **Minimal v2 Changes**: ~15 lines across 4 methods, 12 new adapter methods
5. ✅ **UI Updates**: Adapt to v2 format, show 3 models initially, expandable
6. ✅ **File Rename**: `key_manager.py` → `key_manager_v1.py`, `key_manager_v2.py` → `key_manager.py`

---

## Implementation Order (From FIX_PLAN.md)

1. **Phase 1**: Create custom exceptions and update API clients
2. **Phase 2**: Minimal v2 changes (get_key(), bugs)
3. **Phase 3**: Add adapter methods to v2
4. **Phase 4**: Rename files and update imports
5. **Phase 5**: Update instance management
6. **Phase 6**: Update call sites
7. **Phase 7**: Update UI
8. **Phase 8**: Infrastructure fixes

---

## Status Tracking

- ✅ **FIX_PLAN.md**: Complete and aligned
- ✅ **MIGRATION_AUDIT.md**: Complete and aligned
- ⚠️ **IMPLEMENTATION_PLAN.md**: Marked as superseded
- ✅ **CRITICAL_ISSUES.md**: Complete
- ✅ **MIGRATION_INDEX.md**: This document

---

## Questions?

If you find inconsistencies between documents:
1. **FIX_PLAN.md** takes precedence for implementation details
2. **MIGRATION_AUDIT.md** takes precedence for architecture/design decisions
3. Update this index if you make changes

