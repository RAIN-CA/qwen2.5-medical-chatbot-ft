from typing import Any, Dict, List, Optional


def make_record(
    dataset_name: str,
    domain: str,
    task_type: str,
    system_prompt: str,
    user_text: str,
    assistant_text: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "dataset": dataset_name,
        "domain": domain,
        "task_type": task_type,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
        "metadata": metadata or {},
    }
