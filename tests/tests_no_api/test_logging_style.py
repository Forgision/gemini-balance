from pathlib import Path


def test_no_logger_exception_usage():
    project_root = Path(__file__).resolve().parents[2]
    violations = []
    for py_file in (project_root / "app").rglob("*.py"):
        try:
            text = py_file.read_text(encoding="utf-8")
        except OSError:
            continue
        if "logger.exception(" in text:
            violations.append(py_file.relative_to(project_root))
    assert not violations, f"Disallowed logger.exception usage found in: {violations}"

