import json
from pathlib import Path

import yaml


class KnowledgeBase:
    """Read-only lookup. KB code and actual feature key remain distinct."""
    def __init__(self, path: Path, key_map_path: Path):
        self.entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.key_map: dict[str, str] = yaml.safe_load(key_map_path.read_text(encoding="utf-8"))

    def lookup(self, diagnostic_items) -> list[dict]:
        requested = set(diagnostic_items)
        return [dict(entry) for entry in self.entries if entry["diagnostic_item"] in requested]

