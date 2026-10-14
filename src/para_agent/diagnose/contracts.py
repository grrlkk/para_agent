"""Read the existing key definitions without loading models or changing legacy files."""
import ast
import csv
from pathlib import Path

from para_agent.artifacts import file_hash


def read_legacy_contract(package_path: str | Path) -> dict:
    package = Path(package_path) / "essay_scoring_llm"
    rubric_file = package / "rubrics.py"
    values = {}
    for node in ast.parse(rubric_file.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
            if key in {"RUBRIC_KEYS", "RUBRIC_NAMES_KO", "FEAK_NAMES"}:
                values[key] = ast.literal_eval(node.value)
    scaler = package / "feak_feature/korcat_core/apps/cohesion/essay_scoring/features/scaler_notlabel.csv"
    with scaler.open(encoding="utf-8-sig", newline="") as handle:
        feature_keys = [row["feature"] for row in csv.DictReader(handle)]
    if feature_keys != values["FEAK_NAMES"]:
        raise ValueError("Legacy extractor scaler and FEAK_NAMES disagree; do not guess feature keys")
    if len(set(feature_keys)) != 29 or len(set(values["RUBRIC_KEYS"])) != 8:
        raise ValueError("Legacy rubric/feature key count changed")
    return {"rubric_keys": values["RUBRIC_KEYS"], "rubric_names_ko": values["RUBRIC_NAMES_KO"],
            "feature_keys": feature_keys, "sources": {
                str(rubric_file): file_hash(rubric_file), str(scaler): file_hash(scaler)}}
