from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.connection import get_db
from app.database.services import get_all_usage_stats

router = APIRouter(prefix="/usage_stats")


@router.get("/")
async def get_usage_stats(session: AsyncSession = Depends(get_db)):
    """Get all usage statistics."""
    return await get_all_usage_stats(session)
