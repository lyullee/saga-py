# SAGA Python verification — 2026-10-12

The declared Python dependencies were installed from `pyproject.toml` using
the editable development install, and the full Python test suite was executed
from the repository root.

```text
python -m pip install -e ".[dev]"
python -m pip check
python -m pytest -q tests_python
```

Results:

- `pip check`: no broken requirements
- `tests_python`: **204 passed, 0 failed** in 22.54 s
- Python: 3.14.8 on Windows 11
- Repository commit at verification: `4e1b7ca`

The run emitted deprecation warnings from the current Python 3.14,
pytest-asyncio and Starlette/FastAPI compatibility layer. These warnings did
not change the pass/fail result and are retained here so the verification is
reproducible rather than silently presenting a warning-free run.
