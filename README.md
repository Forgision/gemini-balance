# Gemini Balance - Gemini API Proxy and Load Balancer

<p align="center">
  <a href="https://trendshift.io/repositories/13692" target="_blank">
    <img src="https://trendshift.io/api/badge/repositories/13692" alt="snailyp%2Fgemini-balance | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>
  </a>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg" alt="Python"></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-0.100%2B-green.svg" alt="FastAPI"></a>
  <a href="https://www.uvicorn.org/"><img src="https://img.shields.io/badge/Uvicorn-running-purple.svg" alt="Uvicorn"></a>
  <a href="https://t.me/+soaHax5lyI0wZDVl"><img src="https://img.shields.io/badge/Telegram-Group-blue.svg?logo=telegram" alt="Telegram Group"></a>
</p>

> ⚠️ **Important Statement**: This project is licensed under the [CC BY-NC 4.0](LICENSE) license, and **any form of commercial resale service is prohibited**.
> I have never sold services on any platform. If you encounter any sales, it is a resale. Please do not be deceived.

---

## 📖 Project Introduction

**Gemini Balance** is an application built with Python FastAPI, designed to provide proxy and load balancing for the Google Gemini API. It allows you to manage multiple Gemini API Keys and implement key polling, authentication, model filtering, and status monitoring with simple configuration. In addition, the project also integrates image generation and various image hosting upload functions, and supports proxying in the OpenAI API format.

<details>
<summary>📂 View Project Structure</summary>

```plaintext
app/
├── config/       # Configuration management
├── core/         # Core application logic (FastAPI instance creation, middleware, etc.)
├── database/     # Database models and connections
├── domain/       # Business domain objects
├── exception/    # Custom exceptions
├── handler/      # Request handlers
├── log/          # Logging configuration
├── main.py       # Application entry point
├── middleware/   # FastAPI middleware
├── router/       # API routes (Gemini, OpenAI, status page, etc.)
├── scheduler/    # Scheduled tasks (e.g., key status checks)
├── service/      # Business logic services (chat, key management, statistics, etc.)
├── static/       # Static files (CSS, JS)
├── templates/    # HTML templates (e.g., key status page)
└── utils/        # Utility functions
```
</details>

---

## ✨ Feature Highlights

*   **Multi-Key Load Balancing**: Supports configuring multiple Gemini API Keys (`API_KEYS`) for automatic sequential polling, improving availability and concurrency.
*   **Visual Configuration Takes Effect Immediately**: After modifying the configuration through the management backend, it takes effect without restarting the service.
    ![Configuration Panel](files/image4.png)
*   **Dual Protocol API Compatibility**: Supports forwarding CHAT API requests in both Gemini and OpenAI formats.
    *   OpenAI Base URL: `http://localhost:8000(/hf)/v1`
    *   Gemini Base URL: `http://localhost:8000(/gemini)/v1beta`
*   **Image and Text Dialogue and Image Editing**: Supports models for image and text dialogue and image editing through `IMAGE_MODELS` configuration. When calling, use the `configured-model-image` model name.
    ![Dialogue Image Generation](files/image6.png)
    ![Modify Image](files/image7.png)
*   **Web Search**: Supports models for web search through `SEARCH_MODELS` configuration. When calling, use the `configured-model-search` model name.
    ![Web Search](files/image8.png)
*   **Key Status Monitoring**: Provides a `/keys_status` page (requires authentication) to view the status and usage of each key in real-time.
    ![Monitoring Panel](files/image.png)
*   **Detailed Logging**: Provides detailed error logs for easy troubleshooting.
    ![Call Details](files/image1.png)
    ![Log List](files/image2.png)
    ![Log Details](files/image3.png)
*   **Flexible Key Addition**: Supports batch adding keys through regular expressions `gemini_key` and automatic deduplication.
    ![Add Key](files/image5.png)
*   **Failure Retry and Automatic Disabling**: Automatically handles API request failures, retries (`MAX_RETRIES`), and automatically disables keys after excessive failures (`MAX_FAILURES`), with periodic checks for recovery (`CHECK_INTERVAL_HOURS`).
*   **Comprehensive API Compatibility**:
    *   **Embeddings Interface**: Perfectly adapts to the OpenAI format `embeddings` interface.
    *   **Image Generation Interface**: Adapts the `imagen-3.0-generate-002` model interface to the OpenAI image generation interface format.
*   **Automatic Model List Maintenance**: Automatically obtains and synchronizes the latest model lists from Gemini and OpenAI, compatible with the New API.
*   **Proxy Support**: Supports configuring HTTP/SOCKS5 proxies (`PROXIES`) for use in special network environments.
*   **Docker Support**: Provides Docker images for AMD and ARM architectures for quick deployment.
    *   Image Address: `ghcr.io/snailyp/gemini-balance:latest`

---

## 🚀 Quick Start

### Method 1: Use Docker Compose (Recommended)

This is the most recommended deployment method, allowing you to start the application and database with a single command.

1.  **Download `docker-compose.yml`**:
    Get the `docker-compose.yml` file from the project repository.
2.  **Prepare the `.env` file**:
    Copy `.env.example` to `.env` and modify the configuration as needed. Pay special attention that `DATABASE_TYPE` should be set to `mysql` and the `MYSQL_*` related configurations should be filled in.
3.  **Start the service**:
    In the directory where `docker-compose.yml` and `.env` are located, run the following command:
    ```bash
    docker-compose up -d
    ```
    This command will start the `gemini-balance` application and the `mysql` database in the background.

### Method 2: Use Docker Command

1.  **Pull the image**:
    ```bash
    docker pull ghcr.io/snailyp/gemini-balance:latest
    ```
2.  **Prepare the `.env` file**:
    Copy `.env.example` to `.env` and modify the configuration as needed.
3.  **Run the container**:
    ```bash
    docker run -d -p 8000:8000 --name gemini-balance \
    -v ./data:/app/data \
    --env-file .env \
    ghcr.io/snailyp/gemini-balance:latest
    ```
    *   `-d`: Run in the background.
    *   `-p 8000:8000`: Map the container's port 8000 to the host.
    *   `-v ./data:/app/data`: Mount a data volume to persist SQLite data and logs.
    *   `--env-file .env`: Load the environment variable configuration file.

### Method 3: Run Locally (for Development)

1.  **Clone the repository and install dependencies**:
    ```bash
    git clone https://github.com/snailyp/gemini-balance.git
    cd gemini-balance
    pip install -r requirements.txt
    ```
2.  **Configure environment variables**:
    Copy `.env.example` to `.env` and modify the configuration as needed.
3.  **Start the application**:
    ```bash
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    ```
    After the application starts, access `http://localhost:8000`.

---

## ⚙️ API Endpoints

### Gemini API Format (`/gemini/v1beta`)

This endpoint forwards requests directly to the official Gemini API format endpoint and does not include advanced features.

*   `GET /models`: List available Gemini models.
*   `POST /models/{model_name}:generateContent`: Generate content.
*   `POST /models/{model_name}:streamGenerateContent`: Stream content generation.

### OpenAI API Format

#### Compatible with huggingface (HF) format

If you need to use advanced features (such as fake streaming output), use this endpoint.

*   `GET /hf/v1/models`: List models.
*   `POST /hf/v1/chat/completions`: Chat completions.
*   `POST /hf/v1/embeddings`: Create text embeddings.
*   `POST /hf/v1/images/generations`: Generate images.

#### Standard OpenAI Format

This endpoint forwards directly to the official OpenAI compatible API format endpoint and does not include advanced features.

*   `GET /openai/v1/models`: List models.
*   `POST /openai/v1/chat/completions`: Chat completions (recommended for faster speed and to prevent truncation).
*   `POST /openai/v1/embeddings`: Create text embeddings.
*   `POST /openai/v1/images/generations`: Generate images.

---

<details>
<summary>📋 View Full Configuration List</summary>

| Configuration Item | Description | Default Value |
| :--- | :--- | :--- |
| **Database Configuration** | | |
| `DATABASE_TYPE` | Database type: `mysql` or `sqlite` | `mysql` |
| `SQLITE_DATABASE` | Required when using `sqlite`, SQLite database file path | `default_db` |
| `MYSQL_HOST` | Required when using `mysql`, MySQL database host address | `localhost` |
| `MYSQL_SOCKET` | Optional, MySQL database socket address | `/var/run/mysqld/mysqld.sock` |
| `MYSQL_PORT` | Required when using `mysql`, MySQL database port | `3306` |
| `MYSQL_USER` | Required when using `mysql`, MySQL database username | `your_db_user` |
| `MYSQL_PASSWORD` | Required when using `mysql`, MySQL database password | `your_db_password` |
| `MYSQL_DATABASE` | Required when using `mysql`, MySQL database name | `defaultdb` |
| **API Related Configuration** | | |
| `API_KEYS` | **Required**, list of Gemini API keys for load balancing | `[]` |
| `ALLOWED_TOKENS` | **Required**, list of allowed access tokens | `[]` |
| `AUTH_TOKEN` | Super administrator token, if not filled, the first one from `ALLOWED_TOKENS` is used | `sk-123456` |
| `TEST_MODEL` | Model used to test key availability | `gemini-2.5-flash-lite` |
| `IMAGE_MODELS` | List of models that support drawing functions | `["gemini-2.0-flash-exp", "gemini-2.5-flash-image-preview"]` |
| `SEARCH_MODELS` | List of models that support search functions | `["gemini-2.5-flash","gemini-2.5-pro"]` |
| `FILTERED_MODELS` | List of disabled models | `[]` |
| `TOOLS_CODE_EXECUTION_ENABLED` | Whether to enable the code execution tool | `false` |
| `SHOW_SEARCH_LINK` | Whether to display the search result link in the response | `true` |
| `SHOW_THINKING_PROCESS` | Whether to show the model's thinking process | `true` |
| `THINKING_MODELS` | List of models that support thinking functions | `[]` |
| `THINKING_BUDGET_MAP` | Thinking function budget map (model name: budget value) | `{}` |
| `URL_NORMALIZATION_ENABLED` | Whether to enable smart routing mapping | `false` |
| `URL_CONTEXT_ENABLED` | Whether to enable URL context understanding | `false` |
| `URL_CONTEXT_MODELS` | List of models that support URL context understanding | `[]` |
| `BASE_URL` | Gemini API base URL | `https://generativelanguage.googleapis.com/v1beta` |
| `MAX_FAILURES` | Maximum number of failures allowed for a single key | `3` |
| `MAX_RETRIES` | Maximum number of retries for API request failures | `3` |
| `CHECK_INTERVAL_HOURS` | Disabled key recovery check interval (hours) | `1` |
| `TIMEZONE` | Timezone used by the application | `Asia/Shanghai` |
| `TIME_OUT` | Request timeout (seconds) | `300` |
| `PROXIES` | List of proxy servers (e.g., `http://user:pass@host:port`) | `[]` |
| **Logging and Security** | | |
| `LOG_LEVEL` | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` | `INFO` |
| `ERROR_LOG_RECORD_REQUEST_BODY` | Whether to record the request body in error logs (may contain sensitive information) | `false` |
| `AUTO_DELETE_ERROR_LOGS_ENABLED` | Whether to automatically delete error logs | `true` |
| `AUTO_DELETE_ERROR_LOGS_DAYS` | Error log retention days | `7` |
| `AUTO_DELETE_REQUEST_LOGS_ENABLED`| Whether to automatically delete request logs | `false` |
| `AUTO_DELETE_REQUEST_LOGS_DAYS` | Request log retention days | `30` |
| `SAFETY_SETTINGS` | Content safety threshold (JSON string) | `[{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"}, ...]` |
| **TTS Related** | | |
| `TTS_MODEL` | TTS model name | `gemini-2.5-flash-preview-tts` |
| `TTS_VOICE_NAME` | TTS voice name | `Zephyr` |
| `TTS_SPEED` | TTS speech rate | `normal` |
| **Image Generation Related** | | |
| `PAID_KEY` | Paid API Key for advanced features like image generation | `your-paid-api-key` |
| `CREATE_IMAGE_MODEL` | Image generation model | `imagen-3.0-generate-002` |
| `UPLOAD_PROVIDER` | Image upload provider: `smms`, `picgo`, `cloudflare_imgbed`, `aliyun_oss` | `smms` |
| `OSS_ENDPOINT` | Alibaba Cloud OSS public endpoint | `oss-cn-shanghai.aliyuncs.com` |
| `OSS_ENDPOINT_INNER` | Alibaba Cloud OSS internal endpoint (for intranet access within the same VPC) | `oss-cn-shanghai-internal.aliyuncs.com` |
| `OSS_ACCESS_KEY` | Alibaba Cloud AccessKey ID | `LTAI5txxxxxxxxxxxxxxxx` |
| `OSS_ACCESS_KEY_SECRET` | Alibaba Cloud AccessKey Secret | `yXxxxxxxxxxxxxxxxxxxxxx` |
| `OSS_BUCKET_NAME` | Alibaba Cloud OSS Bucket name | `your-bucket-name` |
| `OSS_REGION` | Alibaba Cloud OSS Region | `cn-shanghai` |
| `SMMS_SECRET_TOKEN` | API Token for SM.MS image hosting | `your-smms-token` |
| `PICGO_API_KEY` | API Key for [PicoGo](https://www.picgo.net/) image hosting | `your-picogo-apikey` |
| `PICGO_API_URL` | API server address for [PicoGo](https://www.picgo.net/) | `https://www.picgo.net/api/1/upload` |
| `CLOUDFLARE_IMGBED_URL` | Upload address for [CloudFlare](https://github.com/MarSeventh/CloudFlare-ImgBed) image hosting | `https://xxxxxxx.pages.dev/upload` |
| `CLOUDFLARE_IMGBED_AUTH_CODE`| Authentication key for CloudFlare image hosting | `your-cloudflare-imgber-auth-code` |
| `CLOUDFLARE_IMGBED_UPLOAD_FOLDER`| Upload folder path for CloudFlare image hosting | `""` |
| **Stream Optimizer Related** | | |
| `STREAM_OPTIMIZER_ENABLED` | Whether to enable stream output optimization | `false` |
| `STREAM_MIN_DELAY` | Minimum delay for stream output | `0.016` |
| `STREAM_MAX_DELAY` | Maximum delay for stream output | `0.024` |
| `STREAM_SHORT_TEXT_THRESHOLD`| Short text threshold | `10` |
| `STREAM_LONG_TEXT_THRESHOLD` | Long text threshold | `50` |
| `STREAM_CHUNK_SIZE` | Stream output chunk size | `5` |
| **Fake Stream Related** | | |
| `FAKE_STREAM_ENABLED` | Whether to enable fake streaming | `false` |
| `FAKE_STREAM_EMPTY_DATA_INTERVAL_SECONDS` | Interval in seconds for sending heartbeat empty data during fake streaming | `5` |

</details>

---

## 🤝 Contributing

Contributions are welcome by submitting Pull Requests or Issues.

[![Contributors](https://contrib.rocks/image?repo=snailyp/gemini-balance)](https://github.com/snailyp/gemini-balance/graphs/contributors)

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=snailyp/gemini-balance&type=Date)](https://star-history.com/#snailyp/gemini-balance&Date)

## 🎉 Special Thanks

*   [PicGo](https://www.picgo.net/)
*   [SM.MS](https://smms.app/)
*   [CloudFlare-ImgBed](https://github.com/MarSeventh/CloudFlare-ImgBed)

## 💖 Friendly Projects

*   **[OneLine](https://github.com/chengtx809/OneLine)** by [chengtx809](https://github.com/chengtx809) - An AI-driven tool for generating timelines of hot events.

## 🎁 Project Support

If you find this project helpful, you can consider supporting me on [爱发电](https://afdian.com/a/snaily).

## License

This project is licensed under the [CC BY-NC 4.0](LICENSE) (Attribution-NonCommercial) license.


## Sponsors

Special thanks to [DigitalOcean](https://m.do.co/c/b249dd7f3b4c) for providing stable and reliable cloud infrastructure support for this project.

<a href="https://m.do.co/c/b249dd7f3b4c">
  <img src="files/dataocean.svg" alt="DigitalOcean Logo" width="200"/>
</a>

The CDN acceleration and security protection for this project are sponsored by [Tencent EdgeOne](https://edgeone.ai/?from=github).

<a href="https://edgeone.ai/?from=github">
  <img src="https://edgeone.ai/media/34fe3a45-492d-4ea4-ae5d-ea1087ca7b4b.png" alt="EdgeOne Logo" width="200"/>
</a>
