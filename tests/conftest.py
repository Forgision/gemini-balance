pytest_plugins = [
    "tests.fixtures.fixture_base",
    "tests.fixtures.fixture_auth",
    "tests.fixtures.fixture_db_simple",
    "tests.fixtures.fixture_mocks",
    "tests.fixtures.fixture_app_simple",
    "tests.fixtures.mock_api", # Keep existing mock_api as it was used by simple tests
]
