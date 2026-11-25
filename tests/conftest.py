from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def keys_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every test gets its own empty key file; the server's saved keys are never touched."""
    path = tmp_path / "provider_keys.json"
    monkeypatch.setenv("AI_KEYS_FILE", str(path))
    return path
