"""Request and response of the retrieval benchmark an admin runs."""

from datetime import date
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

MAX_CASES = 1000
NONE = "none"  # the admin page's "not used" choice

Status = Literal["queued", "running", "done", "failed", "interrupted"]


class EvalCase(BaseModel):
    """One question of the set; extra dataset fields (seed, seed_title) are ignored."""

    id: str | None = None
    category: str = "general"
    question: str = Field(min_length=1, max_length=2000)
    # Keywords for the `like` strategy; the question itself when not given.
    query: str | None = Field(default=None, max_length=200)
    date_from: date
    date_to: date
    relevant: list[str] = Field(min_length=1)
    irrelevant: list[str] = []
    evidence: list[str] = []

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.date_from > self.date_to:
            raise ValueError("date_from must not be after date_to")
        if set(self.relevant) & set(self.irrelevant):
            raise ValueError("an article cannot be both relevant and irrelevant")
        return self


class EvalRequest(BaseModel):
    dataset: str = Field(min_length=1, max_length=200)
    # Ids of configs/models.yaml, and the Jev model for the decision.
    embedding: str
    rerank: str = NONE
    decision: str = NONE
    cases: list[EvalCase] = Field(min_length=1, max_length=MAX_CASES)

    @model_validator(mode="after")
    def _number_cases(self) -> Self:
        for number, case in enumerate(self.cases, start=1):
            case.id = case.id or f"q{number}"
        if len({case.id for case in self.cases}) != len(self.cases):
            raise ValueError("case ids must be unique")
        return self


class Progress(BaseModel):
    done: int = 0
    total: int = 0


class ModelUsage(BaseModel):
    model: str
    calls: int
    input_tokens: int
    output_tokens: int
    estimated: bool = False  # some token counts were estimated from the text length
    cost_usd: float | None  # None: the model has no price


class Spend(BaseModel):
    cost_usd: float | None = 0  # None: a model used has no price
    estimated: bool = False
    usage: list[ModelUsage] = []


class StrategyResult(BaseModel):
    id: str
    overall: dict[str, float | int | None]
    byCategory: dict[str, dict[str, float | int | None]]
    # As if the strategy ran alone: answers it took from a cache count in full.
    seconds: float | None = None
    seconds_per_case: float | None = None
    spend: Spend | None = None


class NotRun(BaseModel):
    id: str
    reason: str


class EvalRun(BaseModel):
    id: str
    status: Status
    created_at: str
    finished_at: str | None = None
    dataset: str
    embedding: str
    rerank: str
    decision: str
    progress: Progress = Progress()
    # What the run really took and paid; None until it ends (and for runs made before).
    seconds: float | None = None
    load_seconds: float | None = None  # reading the chunks and building BM25
    spend: Spend | None = None
    results: list[StrategyResult] = []
    skipped: list[NotRun] = []  # not possible with the chosen models or setup
    failed: list[NotRun] = []  # started and stopped on an error
    error: str | None = None


class EvalRunList(BaseModel):
    runs: list[EvalRun]


class Option(BaseModel):
    id: str
    configured: bool  # its API key is saved


class EvalOptions(BaseModel):
    embeddings: list[Option]
    reranks: list[Option]
    decisions: list[Option]
