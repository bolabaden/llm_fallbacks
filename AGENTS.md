# AGENTS.md

## Cursor Cloud specific instructions

### Project overview
`llm-fallbacks` is a Python library for managing LLM API fallbacks on top of LiteLLM. It is a pure Python package (no web services, no Docker, no databases). Source lives in `src/llm_fallbacks/`, tests in `tests/`.

### Environment variable requirement
The library's `config.py` module initializes an OpenRouter `CustomProviderConfig` with `api_key_required=True` at **import time**. You must set `OPENROUTER_API_KEY` (any non-empty value works for local dev/testing) before importing `llm_fallbacks` or running tests. Example:

```
OPENROUTER_API_KEY=dummy pytest tests/
```

### Common commands
See `README.md` and `.github/workflows/python-package.yml` for canonical commands. Quick reference:

- **Lint:** `ruff check .`
- **Format check:** `black --check .`
- **Type check:** `mypy .`
- **Tests:** `OPENROUTER_API_KEY=dummy pytest --cov=llm_fallbacks tests/ -v`
- **Build:** `python3 -m build`

### PATH note
Dev tools (`ruff`, `black`, `mypy`, `pytest`, etc.) are installed to `~/.local/bin`. The update script ensures this is on `PATH` via `~/.bashrc`, but if commands are not found, run `export PATH="$HOME/.local/bin:$PATH"`.

### Pre-existing lint findings
The codebase has a few pre-existing ruff and mypy findings (e.g. `RUF022` unsorted `__all__`, `RUF069` floating-point comparison, missing stubs for `requests`/`yaml`). These are not introduced by your changes — do not attempt to fix them unless explicitly asked.
