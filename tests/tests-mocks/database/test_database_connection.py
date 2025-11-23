import pytest
import importlib
from unittest.mock import patch, AsyncMock
from app.database import connection
from app.config import config

@pytest.mark.asyncio
async def test_connect_to_db():
    """Test the connect_to_db function."""
    # Mock AsyncSessionLocal to return a mock session
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock()
    
    # Create a mock async context manager
    mock_context_manager = AsyncMock()
    mock_context_manager.__aenter__ = AsyncMock(return_value=mock_session)
    mock_context_manager.__aexit__ = AsyncMock(return_value=None)
    
    with patch("app.database.connection.AsyncSessionLocal", return_value=mock_context_manager):
        await connection.connect_to_db()
        mock_session.execute.assert_called_once()

@pytest.mark.asyncio
async def test_disconnect_from_db():
    """Test the disconnect_from_db function."""
    # Mock engine.dispose()
    with patch("app.database.connection.engine") as mock_engine:
        mock_engine.dispose = AsyncMock()
        await connection.disconnect_from_db()
        mock_engine.dispose.assert_called_once()

def test_database_url_sqlite(monkeypatch):
    """Test the DATABASE_URL for sqlite."""
    monkeypatch.setenv("DATABASE_TYPE", "sqlite")
    monkeypatch.setenv("SQLITE_DATABASE", "test.db")
    importlib.reload(config)
    importlib.reload(connection)
    # Note: The actual implementation uses async driver sqlite+aiosqlite://
    assert connection.DATABASE_URL == "sqlite+aiosqlite:///data/test.db"

def test_database_url_mysql(monkeypatch):
    """Test the DATABASE_URL for mysql."""
    monkeypatch.setenv("DATABASE_TYPE", "mysql")
    monkeypatch.setenv("MYSQL_USER", "user")
    monkeypatch.setenv("MYSQL_PASSWORD", "password")
    monkeypatch.setenv("MYSQL_HOST", "localhost")
    monkeypatch.setenv("MYSQL_PORT", "3306")
    monkeypatch.setenv("MYSQL_DATABASE", "testdb")
    monkeypatch.setenv("MYSQL_SOCKET", "")
    importlib.reload(config)
    importlib.reload(connection)
    # Note: The actual implementation uses async driver mysql+aiomysql://
    assert connection.DATABASE_URL == "mysql+aiomysql://user:password@localhost:3306/testdb"

def test_database_url_mysql_socket(monkeypatch):
    """Test the DATABASE_URL for mysql with a socket."""
    monkeypatch.setenv("DATABASE_TYPE", "mysql")
    monkeypatch.setenv("MYSQL_USER", "user")
    monkeypatch.setenv("MYSQL_PASSWORD", "password")
    monkeypatch.setenv("MYSQL_HOST", "anyhost")
    monkeypatch.setenv("MYSQL_PORT", "3306")
    monkeypatch.setenv("MYSQL_DATABASE", "testdb")
    monkeypatch.setenv("MYSQL_SOCKET", "/var/run/mysqld/mysqld.sock")
    importlib.reload(config)
    importlib.reload(connection)
    # Note: The actual implementation uses async driver mysql+aiomysql://
    assert connection.DATABASE_URL == "mysql+aiomysql://user:password@/testdb?unix_socket=/var/run/mysqld/mysqld.sock"
