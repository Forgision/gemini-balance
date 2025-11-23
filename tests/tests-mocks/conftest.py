from pathlib import Path
import sys


# Add the project root to the Python path to ensure app modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Only load truly global fixtures that are used across all test modules
# Module-specific fixtures should be in their respective conftest.py files
# pytest_plugins removed to fix deprecation warning.
# Plugins should be loaded in the root conftest.py or via pytest.ini if needed globally.
# However, the fixtures mentioned (mocks, auth) seem to be local to this directory structure
# or should be imported directly if they are modules.
# Checking if they are actually needed or if they are automatically discovered.
# If they are in tests/tests-mocks/fixtures, they might need to be imported in this conftest
# or the root conftest.
