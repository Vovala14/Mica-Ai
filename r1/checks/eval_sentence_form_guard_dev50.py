"""Apply the opt-in sentence-form veto to saved dev50 baseline suggestions.

This script uses only the permitted dev50 baseline JSON. It does not run the
decoder or access any evaluation prompt set beyond that saved result.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mica_r1.retrieval_suggest import RetrievalResult, RetrievalSuggestion
from mica_r1.sentence_form_guard import sentence_form_issue, veto_sentence_fragments


CHECKS = Path(__file__).resolve().parents[1] / "runs" / "checks"
SOURCE = CHECKS / "retrieval_dev50_attributed_baseline_sentence.json"
OUTPUT = CHECKS / "retrieval_dev50_attributed_baseline_sentence_form_guard.json"


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    if source["config"]["limit"] != 50 or source["config"]["mode"] != "sentence":
        raise ValueError("expected saved dev50 sentence baseline")
    rows = []
    changed = []
    vetoed_count = 0
    for row in source["rows"]:
        original = RetrievalResult(
            tuple(RetrievalSuggestion(**item) for item in row["suggestions"]),
            row["abstained"], row["reason"],
        )
        filtered = veto_sentence_fragments(original, mode="sentence")
        kept = set(filtered.suggestions)
        vetoed = [
            {"full_text": item.full_text,
             "reason": sentence_form_issue(item.full_text)}
            for item in original.suggestions if item not in kept
        ]
        vetoed_count += len(vetoed)
        old_top = original.suggestions[0].full_text if original.suggestions else None
        new_top = filtered.suggestions[0].full_text if filtered.suggestions else None
        if old_top != new_top:
            changed.append(row["id"])
        rows.append({
            **row,
            "suggestions": [asdict(item) for item in filtered.suggestions],
            "abstained": filtered.abstained,
            "reason": filtered.reason,
            "sentence_form_vetoed": vetoed,
        })
    result = {
        "source_file": SOURCE.name,
        "config": {**source["config"], "post_filter": "sentence_form_guard_v1",
                   "candidate_depth": 3},
        "summary": {
            "prompts": len(rows),
            "usable": sum(not row["abstained"] for row in rows),
            "abstained": sum(row["abstained"] for row in rows),
            "changed_top1": len(changed),
            "changed_top1_ids": changed,
            "vetoed_suggestions": vetoed_count,
        },
        "rows": rows,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), **result["summary"]}, indent=2))
    for row in rows:
        if row["id"] not in changed:
            continue
        top = row["suggestions"][0]["full_text"] if row["suggestions"] else None
        print(f"{row['id']} | {row['prompt']!r} | {top}")


if __name__ == "__main__":
    main()
