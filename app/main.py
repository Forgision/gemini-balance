import uvicorn
from dotenv import load_dotenv

# Load the .env file into environment variables before importing application configuration
load_dotenv()

from app.core.application import create_app  # noqa: E402
from app.log.logger import get_main_logger  # noqa: E402

app = create_app()

if __name__ == "__main__":
    logger = get_main_logger()
    logger.info("Starting application server...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
