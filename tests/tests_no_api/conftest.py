"""
Configuration file for tests_no_api integration tests.
Provides fixtures for in-memory databases, mocked API clients, and test application setup.
"""

pytest_plugins = [
    "tests.fixtures.fixture_db_integration",
    "tests.fixtures.fixture_app_integration",
    "tests.fixtures.fixture_mocks_integration",
]

from tests.fixtures.fixture_consts import TEST_API_KEYS, TEST_VERTEX_API_KEYS, TEST_AUTH_TOKEN, TEST_ALLOWED_TOKENS
