import pytest
import importlib
from unittest.mock import patch, AsyncMock
from app.database import connection
from app.config import config

@pytest.mark.asyncio
async def test_connect_to_db():
    """Test the connect_to_db function."""
    with patch("app.database.connection.database", new_callable=AsyncMock) as mock_database:
        mock_database.is_connected = False
        await connection.connect_to_db()
        mock_database.connect.assert_called_once()

@pytest.mark.asyncio
async def test_disconnect_from_db():
    """Test the disconnect_from_db function."""
    with patch("app.database.connection.database", new_callable=AsyncMock) as mock_database:
        mock_database.is_connected = True
        await connection.disconnect_from_db()
        mock_database.disconnect.assert_called_once()

def test_database_url_sqlite(monkeypatch):
    """Test the DATABASE_URL for sqlite."""
    monkeypatch.setenv("DATABASE_TYPE", "sqlite")
    monkeypatch.setenv("SQLITE_DATABASE", "test.db")
    importlib.reload(config)
    importlib.reload(connection)
    assert connection.DATABASE_URL == "sqlite:///data/test.db"

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
    assert connection.DATABASE_URL == "mysql+pymysql://user:password@localhost:3306/testdb"

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
    assert connection.DATABASE_URL == "mysql+pymysql://user:password@/testdb?unix_socket=/var/run/mysqld/mysqld.sock"
