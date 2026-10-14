"""Load only the first five essays of the unchanged stage 2 pilot."""
import json
from pathlib import Path

from para_agent.artifacts import file_hash
from .loader import load_essays, original_paragraphs


def load_stage3_pilot(manifest_path: Path, cfg):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("stage") != 2 or manifest.get("ok") is not True or manifest.get("essay_count") != 10:
        raise ValueError("Stage 3 requires the successful stage 2 pilot of ten essays")
    if cfg.data.n_essays != 5:
        raise ValueError("Stage 3 live execution is limited to exactly five pilot essays")
    split = Path(manifest["split"])
    if file_hash(split) != manifest["split_sha256"]:
        raise ValueError("Pilot split changed after stage 2")
    essays = load_essays(split, min_paras=cfg.data.min_paras, max_paras=cfg.data.max_paras)
    selected = manifest["sampling"]["selected"]
    if len(essays) != 10 or [e.essay_id for e in essays] != [row["essay_id"] for row in selected]:
        raise ValueError("Pilot selection or ordering changed")
    for name, digest in manifest["audit"]["source_sha256"].items():
        if file_hash(Path(manifest["audit"]["source_paths"][name])) != digest:
            raise ValueError(f"Audited source changed: {name}")
    for essay, row in zip(essays[:5], selected[:5]):
        raw_path = Path(row["raw_source"])
        if file_hash(raw_path) != row["raw_source_sha256"]:
            raise ValueError("Selected raw source changed")
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        if (essay.prompt != raw["essay_question"]["prompt"] or
                essay.paragraphs != original_paragraphs(raw["essay_answer"]["text"])):
            raise ValueError("Pilot text differs from original paragraph boundaries")
    return essays[:5], {"input_file": str(split), "input_sha256": file_hash(split),
                       "split_manifest": str(manifest_path.resolve()),
                       "split_manifest_sha256": file_hash(manifest_path),
                       "selection": "first_five_in_stage2_stratified_order",
                       "essay_ids": [e.essay_id for e in essays[:5]]}
