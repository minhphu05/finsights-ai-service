"""Admin management of provider API keys (stored by app/integrations/key_store.py).

The gateway authenticates the admin. A saved key is not tested here: a wrong one shows up
when it is used. Responses carry the last four characters of a key at most.
"""

import logging

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel

from app.integrations import key_store
from app.integrations.llm.catalog import load_catalog
from app.integrations.providers import get_registry

router = APIRouter(prefix="/admin/keys", tags=["admin"])
logger = logging.getLogger(__name__)

# Jev is not in configs/models.yaml: it has its own module.
JEV_KEY = "typesafe"
MIN_LENGTH = 8
MAX_LENGTH = 512
UNAVAILABLE = "The key file is unavailable."


class KeyInfo(BaseModel):
    provider: str
    configured: bool
    last4: str | None = None
    updated_at: str | None = None


class KeyList(BaseModel):
    keys: list[KeyInfo]


class KeyInput(BaseModel):
    # Length and spaces are checked in the handler: a validation error echoes its input.
    key: str


def known_keys() -> list[str]:
    """Every key name the configured models refer to, in config order."""
    refs = [m.api_key_ref for m in load_catalog().models]
    for provider in get_registry().providers:
        refs.append(provider.api_key_ref)
        if provider.account_id_ref:
            refs.append(provider.account_id_ref)
    names = [ref.partition(":")[2] for ref in refs]
    return list(dict.fromkeys([*names, JEV_KEY]))


def _info(name: str, saved: dict[str, str] | None) -> KeyInfo:
    if not saved:
        return KeyInfo(provider=name, configured=False)
    return KeyInfo(
        provider=name,
        configured=True,
        last4=saved["key"][-4:],
        updated_at=saved.get("updated_at"),
    )


def _require_known(name: str) -> None:
    if name not in known_keys():
        raise HTTPException(status_code=404, detail="Unknown provider.")


@router.get("", response_model=KeyList)
def list_keys() -> KeyList:
    try:
        saved = key_store.entries()
    except key_store.KeyStoreError:
        logger.exception("reading the key file failed")
        raise HTTPException(status_code=503, detail=UNAVAILABLE) from None
    return KeyList(keys=[_info(name, saved.get(name)) for name in known_keys()])


@router.put("/{provider}", response_model=KeyInfo)
def save_key(provider: str, body: KeyInput) -> KeyInfo:
    _require_known(provider)
    key = body.key.strip()
    if not MIN_LENGTH <= len(key) <= MAX_LENGTH or any(c.isspace() for c in key):
        raise HTTPException(
            status_code=422,
            detail=f"The key must be {MIN_LENGTH}-{MAX_LENGTH} characters, no spaces.",
        )
    try:
        key_store.save(provider, key)
        return _info(provider, key_store.entries()[provider])
    except OSError, key_store.KeyStoreError:
        logger.exception("saving a key failed")
        raise HTTPException(status_code=503, detail=UNAVAILABLE) from None


@router.delete("/{provider}", status_code=204)
def delete_key(provider: str) -> Response:
    _require_known(provider)
    try:
        key_store.delete(provider)
    except OSError, key_store.KeyStoreError:
        logger.exception("deleting a key failed")
        raise HTTPException(status_code=503, detail=UNAVAILABLE) from None
    return Response(status_code=204)
