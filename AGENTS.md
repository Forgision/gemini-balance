# Rule file for cursor (**MUST FOLLOW**)

## Structure Guide
- tests from [tests_no_api](./tests/tests_no_api/) are not using mocks/patches/fixture except in memory database creation, patching api client, keymanager, etc. The purpose of these tests are to mimick user like interactions and not fully rely on mocks and patches.
- tests from [tests-mocks](./tests/tests-mocks/) are using mocks, fixtures and patches to test functionality of each file. The purpose of these tests that are classes and function doing what they created for.

## Implementation Rules
- Keep mocks/patches/fixture for tests from [tests_no_api](./tests/tests_no_api/) and [tests-mocks](./tests/tests-mocks/) separated as they have different purpose.
- Don't 

## Rules for executing python file and managing python enviornment
- Use `uv` as python package manager.
- Use `uv sync` command to create a virtual environment and install dependencies.
- Alwasy activate [.venv](./.venv) using `source ./.venv/bin/acitvate` for linux system.
- Use `uv add` command to install python packages.
- Use `uv run` command to run python scripts.
- Use [pyproject.toml](./pyproject.toml) instead of requirements.txt files.

## Rules for unit tests

### fixture classification
- Define global fixtures in [conftest.py](./tests/conftest.py) for use across all tests.
- Define module-specific fixtures in `conftest.py` for tests within a module. e.g fixuture for tests from [services](./tests/tests-mocks/services) puts in [conftest.py](./tests/tests-mocks/services/conftest.py) which are only use for tests of services.
- Define file-specific fixtures within the test file if they are only used there.

### KeyManager test classification
- tests at [test_service_key_key_manager.py](./tests/tests-mocks/services/test_service_key_key_manager.py) file are for [key_manager.py](./app/service/key/key_manager.py).
- tests at [test_service_key_key_manager_v2.py](./tests/tests-mocks/services/test_service_key_key_manager_v2.py) are for [key_manager_v2.py](./app/service/key/key_manager_v2.py).
- Don't mix tests, while creating tests for any of file, put them in their respective file.
- While modify in any file, only modify their respective test file only. e.g. if any modification done in [key_manager_v2.py](./app/service/key/key_manager_v2.py), you have to only modify tests in [test_service_key_key_manager_v2.py](./tests/tests-mocks/services/test_service_key_key_manager_v2.py) test file only, or vice versa.

### General
- [tests_frontend.py](./tests/tests-mocks/test_frontend.py) used for only testsing that frond UI working or not.
- Use setup_module to set up test requirements at the module level.
- Use teardown_module to clean up module-specific resources after tests.
- **Important** always run tests with **timeout**. Without timeout will stuck in infinite running test if there is problem.
- if any test timeout don't consider it as pass, but add debug output point to indetify error.

## Database
- Use sqlalchemy async functionality.
- Always use sqlalchemy 2.0 syntax.


## Current Goal
- Update usage of service based on UsageState model for token and key exhaused statue to new KeyManager.update_usage method from [key_manager.py](./app/service/key/key_manager.py)
<!-- ## Problems in project
- In ClaudeProxyService.create_message on line  -->