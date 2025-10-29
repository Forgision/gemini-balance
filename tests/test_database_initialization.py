import pytest
from unittest.mock import patch, MagicMock
from app.database.initialization import initialize_database, create_tables, import_env_to_settings

def test_initialize_database():
    """Test the initialize_database function."""
    with patch("app.database.initialization.create_tables") as mock_create_tables:
        with patch("app.database.initialization.import_env_to_settings") as mock_import_env_to_settings:
            initialize_database()
            mock_create_tables.assert_called_once()
            mock_import_env_to_settings.assert_called_once()

def test_create_tables():
    """Test the create_tables function."""
    with patch("app.database.initialization.engine") as mock_engine:
        with patch("app.database.initialization.Base") as mock_base:
            create_tables()
            mock_base.metadata.create_all.assert_called_once_with(mock_engine)

@patch("app.database.initialization.dotenv_values")
@patch("app.database.initialization.inspect")
@patch("app.database.initialization.Session")
def test_import_env_to_settings(mock_session, mock_inspect, mock_dotenv_values):
    """Test the import_env_to_settings function."""
    mock_dotenv_values.return_value = {"key1": "value1", "key2": "value2"}
    mock_inspect.return_value.get_table_names.return_value = ["t_settings"]

    # Mock the session and query results
    mock_session_instance = mock_session.return_value.__enter__.return_value
    mock_session_instance.query.return_value.all.return_value = [MagicMock(key="key1", value="old_value")]

    import_env_to_settings()

    mock_session_instance.add.assert_called_once()
    mock_session_instance.commit.assert_called_once()
