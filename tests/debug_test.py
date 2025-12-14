import pytest
from app.config.config import settings


@pytest.mark.asyncio
async def test_debug_settings(test_client, goog_api_key_header):
    with open("debug_output.txt", "w") as f:
        f.write("DEBUG TEST: inside test_debug_settings\n")
        f.write(f"DEBUG TEST: id(settings)={id(settings)}\n")
        f.write(f"DEBUG TEST: settings.AUTH_TOKEN='{settings.AUTH_TOKEN}'\n")
        f.write(f"DEBUG TEST: settings.ALLOWED_TOKENS={settings.ALLOWED_TOKENS}\n")
        f.write(f"DEBUG TEST: header={goog_api_key_header}\n")

        response = test_client.get("/v1beta/models", headers=goog_api_key_header)
        f.write(f"DEBUG TEST: response status={response.status_code}\n")
        f.write(f"DEBUG TEST: response body={response.text}\n")
        f.write(f"DEBUG TEST: response headers={response.headers}\n")

    assert response.status_code == 200
