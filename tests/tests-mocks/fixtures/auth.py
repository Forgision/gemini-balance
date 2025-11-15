import pytest


@pytest.fixture
def mock_verify_auth_token(request, mocker):
    """
    Fixture to mock auth token verification across all routes.
    Tests that need auth mocking should explicitly request this fixture.
    Tests checking unauthorized access should NOT use this fixture.
    """
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
