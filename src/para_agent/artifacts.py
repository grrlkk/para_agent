import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import yaml
from pydantic import BaseModel


def json_text(value) -> str:
    def encode(item):
        if isinstance(item, BaseModel):
            return item.model_dump()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"Not JSON serializable: {type(item).__name__}")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      default=encode)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json_text(value) + "\n")


def append_jsonl(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json_text(value) + "\n")


def create_run(root: Path, project: Path, config: dict, provenance: dict) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run = root / f"{stamp}_{uuid4().hex[:10]}"
    run.mkdir(parents=True, exist_ok=False)
    git = subprocess.run(["git", "-C", str(project), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=False)
    snapshot = {**config, "provenance": {**provenance,
                "git_commit": git.stdout.strip() if git.returncode == 0 else None,
                "git_status": "available" if git.returncode == 0 else "not_a_git_repository"}}
    with (run / "config.yaml").open("x", encoding="utf-8") as handle:
        yaml.safe_dump(snapshot, handle, allow_unicode=True, sort_keys=False)
    (run / "llm_calls.jsonl").touch(exist_ok=False)
    return run

