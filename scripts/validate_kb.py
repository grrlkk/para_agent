#!/usr/bin/env python
"""Validate KB codes against the existing feature definitions; no models or API calls."""
import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from para_agent.config import load_config, project_path
from para_agent.kb.validate import validate_kb


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/default.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    result = validate_kb(project_path(cfg.kb.features), project_path(cfg.kb.feature_key_map), cfg.features.package_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

