from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field

from .schema import Schema

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ModelConfig(Schema):
    provider: str
    model: str
    temperature: float | None = 0
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"] | None = None
    thinking_mode: Literal['adaptive', 'disabled', 'between_tools'] | None = None
    max_output_tokens: int = Field(default=8192, ge=1)
    timeout_s: float = Field(default=180, gt=0)


class LLMConfig(Schema):
    agent: ModelConfig
    judge: ModelConfig
    env_file: str = ".env"


class HeldOutSourceConfig(Schema):
    test_jsonl: str
    train_jsonl: str
    raw_test_dir: str
    audit_manifest: str
    training_evidence: str


class DataConfig(Schema):
    min_paras: int = Field(default=3, ge=1)
    max_paras: int = Field(default=6, ge=1)
    n_essays: int = Field(default=10, ge=1)
    mock_input: str
    seed: int = 42
    split_output: str = "data/splits"
    source: HeldOutSourceConfig | None = None


class ScorerConfig(Schema):
    package_path: str
    backend: str | None = None
    repository_path: str = "/home/chanwoo/FEAK-Agent"
    source_config: str = "/home/chanwoo/FEAK-Agent/verak/v3/config.yaml"
    cache_dir: str = ".cache/scorer"
    metadata_dir: str | None = None


class FeatureConfig(Schema):
    package_path: str
    entrypoint: str
    env_file: str = "/home/chanwoo/essay_scoring_llm/.env"
    timeout_s: float = Field(default=300, gt=0)
    gpu: int | None = Field(default=None, ge=0)


class KBConfig(Schema):
    features: str
    feature_key_map: str


class PathsConfig(Schema):
    prompts: str
    runs: str
    cache: str
    contracts: str
    rubric_diag_map: str
    rubric_unit_map: str
    evaluation_policy: str = 'configs/evaluation_policy.yaml'


class Config(Schema):
    condition: Literal["whole_once", "incremental"] = "incremental"
    use_memory: bool = True
    diag_unit: Literal["para", "cum", "both", "hybrid"] = "both"
    delta_mode: Literal["prev", "cum"] = "prev"
    reviser_input: Literal["x_only", "prev_plus_x"] = "x_only"
    k_issues: int = Field(default=4, ge=1)
    weak_rubrics: int = Field(default=2, ge=1, le=8)
    llm: LLMConfig
    data: DataConfig
    scorer: ScorerConfig
    features: FeatureConfig
    kb: KBConfig
    paths: PathsConfig


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | Path) -> Config:
    config = Config.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    if config.data.min_paras > config.data.max_paras:
        raise ValueError("data.min_paras must not exceed data.max_paras")
    return config

