"""Runtime settings, read from environment variables."""

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RetrievalWeights:
    """Score weights for fact-aware ranking."""

    text: float = 1.0
    fact_match: float = 0.6
    fact_mismatch: float = 0.8


@dataclass(frozen=True)
class Settings:
    """All tunable settings. Override with STEWARD_* environment variables."""

    data_dir: Path = PROJECT_ROOT / "data"
    manifest_path: Path = PROJECT_ROOT / "data" / "manifest.json"
    cache_dir: Path = PROJECT_ROOT / "data" / "cache"
    index_path: Path = PROJECT_ROOT / "data" / "index.sqlite"
    replay_dir: Path = PROJECT_ROOT / "replays"
    eval_dir: Path = PROJECT_ROOT / "eval"
    model_base_url: str = "http://127.0.0.1:8080/v1"
    model_name: str = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    model_api_key: str = ""
    model_timeout_s: float = 180.0
    # Sent as-is in each chat request. The default turns off Qwen3 thinking mode.
    model_extra_body: str = '{"chat_template_kwargs": {"enable_thinking": false}}'
    max_tool_calls: int = 6
    max_model_turns: int = 8
    excerpt_chars: int = 400
    max_passages_per_doc: int = 4
    max_docs_per_passage_call: int = 4
    top_k: int = 5
    max_guideline_citations: int = 8
    weights: RetrievalWeights = field(default_factory=RetrievalWeights)

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings, applying STEWARD_* environment overrides."""
        base = cls()
        index = os.environ.get("STEWARD_INDEX")
        return cls(
            index_path=Path(index) if index else base.index_path,
            model_base_url=os.environ.get("STEWARD_MODEL_BASE_URL", base.model_base_url),
            model_name=os.environ.get("STEWARD_MODEL", base.model_name),
            model_api_key=os.environ.get("STEWARD_API_KEY", base.model_api_key),
            model_timeout_s=float(os.environ.get("STEWARD_MODEL_TIMEOUT", base.model_timeout_s)),
            model_extra_body=os.environ.get("STEWARD_MODEL_EXTRA_BODY", base.model_extra_body),
        )
