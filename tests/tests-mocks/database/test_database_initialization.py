import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from app.database.initialization import initialize_database, create_tables, import_env_to_settings

@pytest.mark.asyncio
async def test_initialize_database():
    """Test the initialize_database function."""
    with patch("app.database.initialization.create_tables", new_callable=AsyncMock) as mock_create_tables:
        with patch("app.database.initialization.import_env_to_settings", new_callable=AsyncMock) as mock_import_env_to_settings:
            await initialize_database()
            mock_create_tables.assert_called_once()
            mock_import_env_to_settings.assert_called_once()

@pytest.mark.asyncio
async def test_create_tables():
    """Test the create_tables function."""
    # Mock engine.begin() to return an async context manager
    mock_conn = AsyncMock()
    mock_conn.run_sync = AsyncMock()
    
    mock_context_manager = AsyncMock()
    mock_context_manager.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_context_manager.__aexit__ = AsyncMock(return_value=None)
    
    with patch("app.database.initialization.engine") as mock_engine:
        mock_engine.begin.return_value = mock_context_manager
        with patch("app.database.initialization.Base") as mock_base:
            await create_tables()
            mock_conn.run_sync.assert_called_once_with(mock_base.metadata.create_all)

@pytest.mark.asyncio
@patch("app.database.initialization.dotenv_values")
@patch("app.database.initialization.AsyncSessionLocal")
@patch("app.database.initialization.engine")
async def test_import_env_to_settings(mock_engine, mock_session_local, mock_dotenv_values):
    """Test the import_env_to_settings function."""
    mock_dotenv_values.return_value = {"key1": "value1", "key2": "value2"}
    
    # Mock engine.begin() for table inspection using run_sync
    mock_conn = AsyncMock()
    # Mock run_sync to return table names directly
    mock_conn.run_sync = AsyncMock(return_value=["t_settings"])
    
    mock_engine_context = AsyncMock()
    mock_engine_context.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine_context.__aexit__ = AsyncMock(return_value=None)
    mock_engine.begin.return_value = mock_engine_context

    # Mock AsyncSessionLocal for database operations
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [MagicMock(key="key1", value="old_value")]
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()
    
    mock_session_context = AsyncMock()
    mock_session_context.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_context.__aexit__ = AsyncMock(return_value=None)
    mock_session_local.return_value = mock_session_context

    await import_env_to_settings()

    mock_session.add.assert_called()
    mock_session.commit.assert_called_once()
