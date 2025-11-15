from pathlib import Path
import sys
from dotenv import load_dotenv

# Add the project root to the Python path to ensure app modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load test environment variables before importing any application modules
# This is crucial to ensure the app is configured for testing
load_dotenv(dotenv_path=PROJECT_ROOT / ".env.test", override=True)