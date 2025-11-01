import pytest


@pytest.fixture(autouse=True)
def mock_verify_auth_token(request, mocker):
    if "no_mock_auth" in request.keywords:
        return

    mocker.patch("app.core.security.verify_auth_token", return_value=True)
    mocker.patch("app.middleware.middleware.verify_auth_token", return_value=True)
    mocker.patch("app.router.config_routes.verify_auth_token", return_value=True)
    mocker.patch("app.router.error_log_routes.verify_auth_token", return_value=True)
    mocker.patch("app.router.key_routes.verify_auth_token", return_value=True)
    mocker.patch("app.router.routes.verify_auth_token", return_value=True)
    mocker.patch("app.router.scheduler_routes.verify_auth_token", return_value=True)
    mocker.patch("app.router.stats_routes.verify_auth_token", return_value=True)
