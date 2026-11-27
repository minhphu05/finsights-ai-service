"""Provider API keys saved by an admin, kept in a local JSON file on the server.

The file is `data/provider_keys.json` (`data/` is git-ignored; override with AI_KEYS_FILE).
It is read on every lookup, so a saved or removed key takes effect at once. Writes replace
the file atomically; the lock covers one process, which is how the service runs. Keys are
stored as given and are not tested here. Never log or return a key.
"""

import json
import os
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILE = _ROOT / "data" / "provider_keys.json"

_lock = threading.Lock()


class KeyStoreError(ValueError):
    """A key cannot be read: not saved, an unknown reference or a broken file."""


def keys_file() -> Path:
    return Path(os.environ.get("AI_KEYS_FILE") or DEFAULT_FILE)


def entries() -> dict[str, dict[str, str]]:
    """name -> {"key": ..., "updated_at": ...} for every saved key."""
    path = keys_file()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except OSError, ValueError:
        # No file content in the message: it holds keys.
        raise KeyStoreError(f"cannot read the key file {path.name}") from None
    if not isinstance(data, dict):
        raise KeyStoreError(f"the key file {path.name} is not a JSON object")
    return data


def resolve(ref: str) -> str:
    """The key `ref` (`key:<name>`) points to; the error never contains a key."""
    scheme, _, name = ref.partition(":")
    if scheme != "key":
        raise KeyStoreError(f"unsupported key reference {scheme!r} (supported: key)")
    value = entries().get(name, {}).get("key", "")
    if not value:
        raise KeyStoreError(f"no API key saved for {name!r}")
    return value


def save(name: str, key: str) -> None:
    with _lock:
        data = entries()
        data[name] = {"key": key, "updated_at": datetime.now(UTC).isoformat()}
        _write(data)


def delete(name: str) -> None:
    with _lock:
        data = entries()
        if data.pop(name, None) is not None:
            _write(data)


def _write(data: dict[str, dict[str, str]]) -> None:
    path = keys_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    # mkstemp creates the file readable by its owner only (mode 600 on POSIX).
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
