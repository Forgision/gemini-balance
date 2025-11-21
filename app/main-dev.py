import os
from pathlib import Path
import uvicorn
from dotenv import load_dotenv
from app.log.logger import get_main_logger  # noqa: E402

# Load the .env file into environment variables before importing application configuration
load_dotenv()

logger = get_main_logger()

# delete existing database
if Path("data/default_db").exists():
    Path("data/default_db").unlink()
    try:
        os.remove("data/default_db")
        logger.info("Database file removed successfully")
    except Exception as e:
        logger.error(f"Failed to remove database file: {e}")
        

from app.core.application import create_app  # noqa: E402

app = create_app()

if __name__ == "__main__":
    logger.info("Starting application server...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
