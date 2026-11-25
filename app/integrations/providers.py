"""Embedding and rerank models as data, from `configs/models.yaml` (or AI_MODELS_FILE).

The file has three sections: `embedding`, `rerank` and `llm` (read by the LLM catalog). An
embedding or rerank model is an entry with its URL and a reference to its API key. The clients
read the `default` of their section, so changing a model or adding a backend is a config
change. Keys never live in the file: `api_key_ref` is `key:<name>`, a key an admin
saved (app/integrations/key_store.py). Never log a resolved key.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.integrations import key_store

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILE = _ROOT / "configs" / "models.yaml"

Kind = Literal["embedding", "rerank"]
KINDS: tuple[Kind, ...] = ("embedding", "rerank")
# Embedding APIs: voyage, and Cloudflare Workers AI (`@cf/...` models)
Dialect = Literal["voyage", "cloudflare"]


class ProviderConfigError(ValueError):
    """A provider config is unusable: unknown key scheme or a key that is not set."""


class ProviderConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: Annotated[str, Field(min_length=1)]
    kind: Kind
    model: Annotated[str, Field(min_length=1)]
    url: Annotated[str, Field(min_length=1)]
    api_key_ref: Annotated[str, Field(pattern=r"^[a-z]+:.+$")]
    # embedding only: the API shape the model speaks, like `dialect` of an LLM
    dialect: Dialect | None = None
    # embedding only: vector size sent to the API (voyage)
    dimensions: Annotated[int, Field(ge=256, le=2048)] | None = None
    # cloudflare only: reference to the account id the `url` template needs
    account_id_ref: Annotated[str, Field(pattern=r"^[a-z]+:.+$")] | None = None

    @model_validator(mode="after")
    def _check(self) -> ProviderConfig:
        if self.kind == "embedding" and self.dialect is None:
            raise ValueError(f"embedding model {self.id!r} needs a dialect")
        if self.dialect == "cloudflare" and self.account_id_ref is None:
            raise ValueError(f"cloudflare model {self.id!r} needs account_id_ref")
        return self

    def api_key(self) -> str:
        """The key this provider uses; the error names the key, never its value."""
        return _resolve(self.api_key_ref)

    def account_id(self) -> str:
        if self.account_id_ref is None:
            raise ProviderConfigError(f"{self.id!r} has no account_id_ref")
        return _resolve(self.account_id_ref)

    def has_key(self) -> bool:
        try:
            self.api_key()
        except ProviderConfigError:
            return False
        return True


class Registry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    providers: tuple[ProviderConfig, ...]
    # kind -> id of the provider the service uses
    defaults: dict[Kind, str]

    @model_validator(mode="after")
    def _check(self) -> Registry:
        ids = [p.id for p in self.providers]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate provider ids: {sorted(ids)}")
        for kind in KINDS:
            chosen = self.defaults.get(kind)
            if chosen is None:
                raise ValueError(f"defaults has no provider for kind {kind!r}")
            found = next((p for p in self.providers if p.id == chosen), None)
            if found is None or found.kind != kind:
                raise ValueError(f"default {kind} provider {chosen!r} is not a {kind}")
        return self

    def default(self, kind: Kind) -> ProviderConfig:
        wanted = self.defaults[kind]
        return next(p for p in self.providers if p.id == wanted)


def _resolve(ref: str) -> str:
    """Read the value `ref` (`key:<name>`) points to; the error never contains it."""
    try:
        return key_store.resolve(ref)
    except key_store.KeyStoreError as exc:
        raise ProviderConfigError(str(exc)) from None


def read_models_file(path: str | Path | None = None) -> dict:
    """The whole models file as a dict; each section is validated by whoever reads it."""
    source = Path(path or os.environ.get("AI_MODELS_FILE") or DEFAULT_FILE)
    return yaml.safe_load(source.read_text(encoding="utf-8"))


@lru_cache
def get_registry() -> Registry:
    data = read_models_file()
    return Registry(
        providers=tuple(
            ProviderConfig(kind=kind, **entry)
            for kind in KINDS
            for entry in data[kind]["models"]
        ),
        defaults={kind: data[kind]["default"] for kind in KINDS},
    )


def get_provider(kind: Kind) -> ProviderConfig:
    return get_registry().default(kind)
