#!/usr/bin/env python3
"""Prepare merged/cleaned CivicFlow SFT data from synthetic JSON sources.

Usage:
  python training/sft/prepare_sft_data.py \
      --in syndata.json --in syndata1.json \
      --out-json training/sft/sft_merged_clean.json \
      --out-jsonl training/sft/sft_merged_clean.jsonl \
      --report training/sft/sft_merge_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

EXPECTED_KEYS = {
    "dialogue_id",
    "turn_id",
    "difficulty",
    "type",
    "system_prompt",
    "task_briefing",
    "current_phase",
    "phase_objective",
    "observation_summary",
    "active_constraints",
    "legal_actions_summary",
    "expert_response",
    "annotation",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _stable_id(row: Dict[str, Any]) -> str:
    base = {
        "system_prompt": row.get("system_prompt", ""),
        "task_briefing": row.get("task_briefing", ""),
        "current_phase": row.get("current_phase", ""),
        "phase_objective": row.get("phase_objective", ""),
        "observation_summary": row.get("observation_summary", {}),
        "active_constraints": row.get("active_constraints", []),
        "legal_actions_summary": row.get("legal_actions_summary", []),
        "expert_response": row.get("expert_response", {}),
    }
    return hashlib.sha256(_canonical_json(base).encode("utf-8")).hexdigest()[:16]


def _extract_json_blocks(raw: str) -> List[Any]:
    """Extract all decodable JSON blocks from text, including concatenated objects."""
    dec = json.JSONDecoder()
    i = 0
    blocks: List[Any] = []
    n = len(raw)
    while i < n:
        while i < n and raw[i] not in "[{":
            i += 1
        if i >= n:
            break
        try:
            obj, end = dec.raw_decode(raw, i)
            blocks.append(obj)
            i = end
        except json.JSONDecodeError:
            i += 1
    return blocks


def _extract_examples(obj: Any) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []

    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict) and EXPECTED_KEYS.issubset(item.keys()):
                examples.append(item)
    elif isinstance(obj, dict):
        # canonical container in syndata.json
        ces = obj.get("corrected_example_set")
        if isinstance(ces, list):
            for item in ces:
                if isinstance(item, dict) and EXPECTED_KEYS.issubset(item.keys()):
                    examples.append(item)

        # also accept dict itself if it is one example
        if EXPECTED_KEYS.issubset(obj.keys()):
            examples.append(obj)

    return examples


def _to_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _clean_row(row: Dict[str, Any], source_file: str, idx: int) -> Optional[Dict[str, Any]]:
    # Basic structural fields
    system_prompt = str(row.get("system_prompt", "")).strip()
    task_briefing = str(row.get("task_briefing", "")).strip()
    current_phase = str(row.get("current_phase", "")).strip()
    phase_objective = str(row.get("phase_objective", "")).strip()

    obs = row.get("observation_summary", {})
    if not isinstance(obs, dict):
        obs = {"raw": obs}

    constraints = [str(x).strip() for x in _to_list(row.get("active_constraints")) if str(x).strip()]
    legal_actions = [str(x).strip() for x in _to_list(row.get("legal_actions_summary")) if str(x).strip()]

    expert_response = row.get("expert_response", {})
    if not isinstance(expert_response, dict):
        expert_response = {"raw": expert_response}

    annotation = row.get("annotation", {})
    if not isinstance(annotation, dict):
        annotation = {"raw": annotation}

    # discard rows that are effectively non-training wrappers
    if not (system_prompt and task_briefing and current_phase and phase_objective):
        return None

    cleaned: Dict[str, Any] = {
        "dialogue_id": str(row.get("dialogue_id", "")).strip() or f"auto_{source_file}_{idx}",
        "turn_id": int(row.get("turn_id", 0)) if str(row.get("turn_id", "")).strip() else 0,
        "difficulty": str(row.get("difficulty", "tiny")).strip().lower() or "tiny",
        "type": str(row.get("type", "planning_step")).strip() or "planning_step",
        "system_prompt": system_prompt,
        "task_briefing": task_briefing,
        "current_phase": current_phase,
        "phase_objective": phase_objective,
        "observation_summary": obs,
        "active_constraints": constraints,
        "legal_actions_summary": legal_actions,
        "expert_response": expert_response,
        "annotation": annotation,
        "source_file": source_file,
    }

    # Build a model-input string for SFT chat-style formatting.
    user_payload = {
        "task_briefing": cleaned["task_briefing"],
        "current_phase": cleaned["current_phase"],
        "phase_objective": cleaned["phase_objective"],
        "observation_summary": cleaned["observation_summary"],
        "active_constraints": cleaned["active_constraints"],
        "legal_actions_summary": cleaned["legal_actions_summary"],
    }
    cleaned["sft_input"] = _canonical_json(user_payload)
    cleaned["sft_output"] = _canonical_json(cleaned["expert_response"])

    cleaned["example_id"] = _stable_id(cleaned)
    return cleaned


def _load_and_extract(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw = path.read_text(encoding="utf-8", errors="replace")

    blocks = _extract_json_blocks(raw)
    extracted: List[Dict[str, Any]] = []

    for b in blocks:
        extracted.extend(_extract_examples(b))

    stats = {
        "file": str(path),
        "bytes": len(raw),
        "json_blocks_found": len(blocks),
        "candidate_examples_found": len(extracted),
    }
    return extracted, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inputs", action="append", required=True, help="Input synthetic JSON file (repeatable)")
    parser.add_argument("--out-json", required=True, help="Merged cleaned dataset JSON path")
    parser.add_argument("--out-jsonl", required=True, help="Merged cleaned dataset JSONL path")
    parser.add_argument("--report", required=True, help="Merge/clean report path")
    args = parser.parse_args()

    raw_rows: List[Tuple[Dict[str, Any], str, int]] = []
    input_stats: List[Dict[str, Any]] = []

    for in_path in args.inputs:
        p = Path(in_path)
        rows, st = _load_and_extract(p)
        input_stats.append(st)
        for i, row in enumerate(rows):
            raw_rows.append((row, p.name, i))

    cleaned_rows: List[Dict[str, Any]] = []
    dropped_invalid = 0

    for row, src, idx in raw_rows:
        cleaned = _clean_row(row, src, idx)
        if cleaned is None:
            dropped_invalid += 1
            continue
        cleaned_rows.append(cleaned)

    # Deduplicate by content-stable hash.
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for row in cleaned_rows:
        key = row["example_id"]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    # Deterministic order for reproducibility.
    deduped.sort(key=lambda r: (r.get("dialogue_id", ""), int(r.get("turn_id", 0)), r["example_id"]))

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(deduped, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # JSONL for direct trainer ingestion.
    out_jsonl = Path(args.out_jsonl)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in deduped:
            rec = {
                "example_id": row["example_id"],
                "messages": [
                    {"role": "system", "content": row["system_prompt"]},
                    {"role": "user", "content": row["sft_input"]},
                    {"role": "assistant", "content": row["sft_output"]},
                ],
                "meta": {
                    "dialogue_id": row["dialogue_id"],
                    "turn_id": row["turn_id"],
                    "difficulty": row["difficulty"],
                    "type": row["type"],
                    "source_file": row["source_file"],
                },
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    diff_counts: Dict[str, int] = {}
    source_counts: Dict[str, int] = {}
    for row in deduped:
        d = row.get("difficulty", "unknown")
        s = row.get("source_file", "unknown")
        diff_counts[d] = diff_counts.get(d, 0) + 1
        source_counts[s] = source_counts.get(s, 0) + 1

    report = {
        "inputs": input_stats,
        "raw_examples_total": len(raw_rows),
        "cleaned_examples_total": len(cleaned_rows),
        "dropped_invalid": dropped_invalid,
        "deduped_examples_total": len(deduped),
        "duplicates_removed": len(cleaned_rows) - len(deduped),
        "difficulty_distribution": diff_counts,
        "source_distribution": source_counts,
        "outputs": {
            "json": str(out_json),
            "jsonl": str(out_jsonl),
        },
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
