"""The final qualitative evaluation must survive an interrupted run."""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from checks.eval_retrieval import load_prompts, save_result


def test_holdout_shape_and_incremental_result(tmp_path):
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps({"purpose": "sealed", "prompts": [
        "The door was open", "The lights went out",
    ]}), encoding="utf-8")
    assert load_prompts(prompts, 1) == [
        {"id": None, "prompt": "The door was open"},
    ]

    output = tmp_path / "result.json"
    row = {"id": None, "prompt": "The door was open", "abstained": False,
           "seconds": 0.5, "suggestions": []}
    save_result(output, {"mode": "short"}, [row])
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["rows"] == [row]
    assert saved["summary"] == {
        "prompts": 1, "usable": 1, "abstained": 0, "total_seconds": 0.5,
    }
    assert not output.with_name(output.name + ".tmp").exists()


def test_jsonl_development_prompts(tmp_path):
    prompts = tmp_path / "dev.jsonl"
    prompts.write_text('{"id":"dev-1","prompt":"A bunch "}\n'
                       '{"id":"dev-2","prompt":"The kids "}\n',
                       encoding="utf-8")
    assert load_prompts(prompts, 2) == [
        {"id": "dev-1", "prompt": "A bunch "},
        {"id": "dev-2", "prompt": "The kids "},
    ]
