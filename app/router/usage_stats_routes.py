from fastapi import APIRouter
from app.database.services import get_all_usage_stats

router = APIRouter(prefix="/usage_stats")


@router.get("/")
async def get_usage_stats():
    """Get all usage statistics."""
    return await get_all_usage_stats()
