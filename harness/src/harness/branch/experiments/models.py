"""Explicit experiment matrix and sampling/statistics contracts."""
from typing import Any, Literal

from pydantic import Field, model_validator

from ...agent.contracts import Contract, RuntimeConfig
from ..episodes.models import EpisodeConfig
from ...providers import DEFAULT_BASE_URLS, PROVIDER_NAMES


class Variant(Contract):
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    model: str = Field(min_length=1)
    #: Backend that serves this variant; 'ollama' runs locally at no token cost.
    provider: Literal["deepseek", "ollama"] = "deepseek"
    #: Resolved from the provider when omitted, so an Ollama variant does not
    #: silently inherit the DeepSeek endpoint.
    base_url: str | None = None
    api_key_env: str = "DEEPSEEK_API_KEY"
    provider_parameters: dict[str, Any] = Field(default_factory=lambda: {"temperature": 0})
    # Opt-in: most providers do not promise seeded sampling.
    model_seed_parameter: Literal["seed"] | None = None
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    skill_dir: str | None = None
    baseline: Literal["full_query", "long_horizon"] = "long_horizon"

    def parameters(self, repeat_seed: int) -> dict:
        params = dict(self.provider_parameters)
        if self.model_seed_parameter:
            params[self.model_seed_parameter] = repeat_seed
        return params

    @model_validator(mode="before")
    @classmethod
    def fill_provider_defaults(cls, data):
        """Default the endpoint to the selected provider's own base URL."""
        if isinstance(data, dict) and not data.get("base_url"):
            provider = data.get("provider") or "deepseek"
            if provider not in PROVIDER_NAMES:
                raise ValueError(
                    f"unknown provider {provider!r}; expected one of "
                    f"{', '.join(PROVIDER_NAMES)}"
                )
            data = {**data, "base_url": DEFAULT_BASE_URLS[provider]}
        return data

    @model_validator(mode="after")
    def offline_state(self):
        if self.runtime.interaction_mode != "offline":
            raise ValueError("experiment interaction mode is selected by mode/baseline")
        if self.model_seed_parameter and self.model_seed_parameter in self.provider_parameters:
            raise ValueError("configure seed through model_seed_parameter and repeat_seeds")
        return self


class ExperimentConfig(Contract):
    version: str = "shopping-experiment-v1"
    mode: Literal["offline", "episode"] = "offline"
    variants: list[Variant] = Field(min_length=2, max_length=20)
    reference: str
    repeat_seeds: list[int] = Field(default_factory=lambda: [11, 22, 33], min_length=1, max_length=100)
    selection_seed: int = 42
    statistics_seed: int = 2026
    bootstrap_samples: int = Field(default=2000, ge=100, le=10000, strict=True)
    confidence: float = Field(default=0.95, gt=0, lt=1, allow_inf_nan=False)
    # Shared resource envelope; runtime-specific constraints remain per variant.
    budget: EpisodeConfig = Field(default_factory=EpisodeConfig)

    @model_validator(mode="after")
    def unique_matrix(self):
        names = [v.name for v in self.variants]
        if len(names) != len(set(names)) or self.reference not in names:
            raise ValueError("variant names must be unique and reference must name a variant")
        if len(self.repeat_seeds) != len(set(self.repeat_seeds)):
            raise ValueError("repeat seeds must be unique")
        if self.mode == "offline" and any(v.runtime.progressive_disclosure for v in self.variants):
            raise ValueError("progressive disclosure requires episode mode")
        return self
