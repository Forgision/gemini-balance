from pathlib import Path
import sys
from dotenv import load_dotenv

# Add the project root to the Python path to ensure app modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load test environment variables before importing any application modules
# This is crucial to ensure the app is configured for testing
load_dotenv(dotenv_path=PROJECT_ROOT / ".env.test", override=True)


# Only load truly global fixtures that are used across all test modules
# Module-specific fixtures should be in their respective conftest.py files
pytest_plugins = [
    "tests.fixtures.mocks",  # Mock fixtures used everywhere
    "tests.fixtures.auth",   # Auth fixtures used everywhere
    # Note: database fixtures are in tests/database/conftest.py
    # Note: app fixtures are in tests/fixtures/app.py (for non-route tests)
    # Note: route fixtures are in tests/routes/conftest.py
]
