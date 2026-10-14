#!/usr/bin/env python
"""Complete local scorer genre metadata from an audited pilot, without model/API calls."""
import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.dont_write_bytecode = True

from para_agent.config import load_config
from para_agent.diagnose.scorer_metadata import prepare_metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/default.yaml")
    parser.add_argument("--split-manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare_metadata(load_config(args.config), args.split_manifest)
    print(json.dumps({key: result[key] for key in ("metadata_dir", "genres_sha256", "base_question_count",
                                                  "added_question_count", "total_question_count", "selected_essay_count")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
