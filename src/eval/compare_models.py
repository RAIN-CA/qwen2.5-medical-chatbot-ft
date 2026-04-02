import argparse
import json
from pathlib import Path

from src.inference.engine import (
    GenerationConfig,
    load_model_and_tokenizer,
    generate_answer,
    unload_model,
)


DEFAULT_QUESTIONS = [
    "What are the common symptoms of diabetes?",
    "What is hypertension?",
    "What is the difference between CT and MRI?",
    "What are common risk factors for heart disease?",
    "What is anemia?",
]


def load_questions(question_file: str | None):
    if question_file is None:
        return DEFAULT_QUESTIONS

    path = Path(question_file)
    if not path.exists():
        raise FileNotFoundError(f"Question file not found: {path}")

    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
        raise ValueError("JSON question file must contain a list of strings.")

    if path.suffix == ".jsonl":
        questions = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                q = item.get("question") or item.get("instruction")
                if q:
                    questions.append(str(q).strip())
        return questions

    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/eval/model_compare.json",
        help="Path to model comparison config JSON.",
    )
    parser.add_argument(
        "--question_file",
        type=str,
        default=None,
        help="Optional file containing evaluation questions.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/model_comparison_results.json",
        help="Where to save comparison results.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="medical",
        help="Prompt domain: medical/general/...",
    )
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    config_path = Path(args.model_config)
    if not config_path.exists():
        raise FileNotFoundError(f"Model config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        model_configs = json.load(f)

    questions = load_questions(args.question_file)

    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
    )

    results = []

    for model_cfg in model_configs:
        print(f"\n===== Loading {model_cfg['name']} =====")

        tokenizer, model = load_model_and_tokenizer(
            base_model=model_cfg["base_model"],
            adapter_path=model_cfg.get("adapter_path"),
            load_in_4bit=bool(model_cfg.get("load_in_4bit", False)),
        )

        model_result = {
            "model_name": model_cfg["name"],
            "base_model": model_cfg["base_model"],
            "adapter_path": model_cfg.get("adapter_path"),
            "load_in_4bit": bool(model_cfg.get("load_in_4bit", False)),
            "domain": args.domain,
            "answers": [],
        }

        for q in questions:
            print(f"Generating for: {q}")
            ans = generate_answer(
                tokenizer=tokenizer,
                model=model,
                query=q,
                domain=args.domain,
                generation_config=gen_cfg,
            )
            model_result["answers"].append(
                {
                    "question": q,
                    "answer": ans,
                }
            )

        results.append(model_result)
        unload_model(tokenizer, model)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
