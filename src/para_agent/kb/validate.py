"""Validate the unchanged KB using the user-confirmed explicit prefix mapping."""
from para_agent.artifacts import file_hash
from para_agent.diagnose.contracts import read_legacy_contract
from para_agent.kb.lookup import KnowledgeBase

REQUIRED_FIELDS = {"code", "feature_name_ko", "diagnostic_item", "problem_pattern", "cause",
                   "suggestion", "problem_direction", "unit_scope", "length_sensitive", "notes"}


def validate_kb(kb_path, key_map_path, package_path):
    contract = read_legacy_contract(package_path)
    kb = KnowledgeBase(kb_path, key_map_path)
    errors = []
    keys = set(contract["feature_keys"])
    codes = [entry.get("code") for entry in kb.entries]
    if len(kb.entries) != 29 or len(set(codes)) != 29:
        errors.append("Expected 29 unique KB codes")
    if set(codes) != set(kb.key_map):
        errors.append("Explicit mapping keys must exactly match KB codes")
    targets = list(kb.key_map.values())
    if set(targets) != keys or len(targets) != len(set(targets)):
        errors.append("Explicit mapping must cover every actual feature key exactly once")
    for index, entry in enumerate(kb.entries, 1):
        if not REQUIRED_FIELDS <= entry.keys():
            errors.append(f"Line {index}: missing required fields {sorted(REQUIRED_FIELDS - entry.keys())}")
        if entry.get("unit_scope") not in {"para", "cum", "both"}:
            errors.append(f"Line {index}: invalid unit_scope")
        if type(entry.get("length_sensitive")) is not bool:
            errors.append(f"Line {index}: length_sensitive must be boolean")
        if entry.get("problem_direction") not in {"low", "high", "both"}:
            errors.append(f"Line {index}: invalid problem_direction")
    return {"ok": not errors, "errors": errors, "kb_count": len(kb.entries),
            "actual_feature_count": len(keys), "direct_code_match_count": sum(code in keys for code in codes),
            "mapped_code_match_count": sum(kb.key_map.get(code) in keys for code in codes),
            "aliases": {code: key for code, key in kb.key_map.items() if code != key},
            "kb_sha256": file_hash(kb_path), "key_map_sha256": file_hash(key_map_path),
            "legacy_sources": contract["sources"]}
