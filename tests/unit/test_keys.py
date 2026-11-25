from pathlib import Path

from fastapi.testclient import TestClient

from app.integrations import key_store
from app.main import app

KEY = "sk-secret-key-for-tests-WXYZ"

client = TestClient(app)


def _entry(name: str) -> dict:
    keys = client.get("/admin/keys").json()["keys"]
    return next(k for k in keys if k["provider"] == name)


def test_lists_every_configured_provider_without_keys() -> None:
    keys = client.get("/admin/keys").json()["keys"]
    assert {k["provider"] for k in keys} == {
        "anthropic",
        "openai",
        "deepseek",
        "voyage",
        "cloudflare",
        "cloudflare_account_id",
        "typesafe",
    }
    assert not any(k["configured"] for k in keys)


def test_save_stores_the_key_and_returns_only_its_end(keys_file: Path) -> None:
    response = client.put("/admin/keys/deepseek", json={"key": f"  {KEY}  "})
    assert response.status_code == 200
    assert response.json()["last4"] == "WXYZ"
    assert KEY not in response.text
    assert KEY not in client.get("/admin/keys").text
    assert key_store.resolve("key:deepseek") == KEY
    assert list(keys_file.parent.iterdir()) == [keys_file]  # no temporary file left


def test_save_replaces_and_delete_removes() -> None:
    client.put("/admin/keys/voyage", json={"key": KEY})
    client.put("/admin/keys/voyage", json={"key": "pa-another-key-1234"})
    assert _entry("voyage")["last4"] == "1234"
    assert client.delete("/admin/keys/voyage").status_code == 204
    assert _entry("voyage")["configured"] is False


def test_unknown_provider_is_rejected() -> None:
    assert client.put("/admin/keys/nope", json={"key": KEY}).status_code == 404
    assert client.delete("/admin/keys/nope").status_code == 404


def test_bad_key_is_rejected_without_echoing_it() -> None:
    for bad in ("short", "has a space in the middle", "x" * 513):
        response = client.put("/admin/keys/openai", json={"key": bad})
        assert response.status_code == 422
        assert bad not in response.text
    assert _entry("openai")["configured"] is False


def test_broken_file_is_reported_and_not_overwritten(keys_file: Path) -> None:
    keys_file.write_text("not json", encoding="utf-8")
    assert client.get("/admin/keys").status_code == 503
    assert client.put("/admin/keys/openai", json={"key": KEY}).status_code == 503
    assert keys_file.read_text(encoding="utf-8") == "not json"


def test_health_reports_missing_keys_without_calling_providers() -> None:
    body = client.get("/health/providers").json()
    assert body["status"] == "not_configured"
    assert body["jev"]["keys"] == ["typesafe"]
    assert body["llm"]["deepseek"]["keys"] == ["deepseek"]
    voyage = next(p for p in body["embedding"] if p["id"] == "voyage-4")
    assert voyage["status"] == "not_configured"
    assert voyage["keys"] == ["voyage"]
