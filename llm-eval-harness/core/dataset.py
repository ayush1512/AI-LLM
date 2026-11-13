import json
import io
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

@dataclass
class EvalCase:
    case_id: str
    input_prompt: str
    expected_output: Optional[str] = None
    context: Optional[str] = None
    task_type: str = "general_qa"  # "qa", "summarization", "json_extraction", "safety"
    metadata: Dict[str, Any] = field(default_factory=dict)

def load_dataset_from_jsonl(file_source: Any) -> List[EvalCase]:
    """Loads evaluation dataset from a file path, string, or file-like object."""
    cases: List[EvalCase] = []
    
    if isinstance(file_source, str) and "\n" not in file_source and (file_source.endswith(".jsonl") or file_source.endswith(".json")):
        with open(file_source, "r", encoding="utf-8") as f:
            lines = f.readlines()
    elif isinstance(file_source, bytes):
        lines = file_source.decode("utf-8").splitlines()
    elif isinstance(file_source, str):
        lines = file_source.splitlines()
    else:
        lines = file_source.readlines()

    for idx, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            cases.append(
                EvalCase(
                    case_id=str(data.get("case_id", f"case-{idx}")),
                    input_prompt=data.get("input_prompt", data.get("input", "")),
                    expected_output=data.get("expected_output", data.get("reference")),
                    context=data.get("context"),
                    task_type=data.get("task_type", "general_qa"),
                    metadata=data.get("metadata", {})
                )
            )
        except json.JSONDecodeError:
            continue

    return cases

def save_dataset_to_jsonl(cases: List[EvalCase], file_path: str):
    with open(file_path, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(asdict(case)) + "\n")

