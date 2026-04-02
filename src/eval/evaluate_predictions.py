import argparse
import json
from pathlib import Path
from collections import defaultdict

import evaluate
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def read_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, required=True)
    parser.add_argument("--report_file", type=str, required=True)
    args = parser.parse_args()

    data = read_jsonl(args.pred_file)

    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    predictions = [x["prediction"] for x in data]
    references = [x["reference"] for x in data]

    rouge_scores = rouge.compute(predictions=predictions, references=references)

    bleu_scores = []
    for pred, ref in zip(predictions, references):
        try:
            score = bleu.compute(
                predictions=[pred],
                references=[[ref]],
            )["bleu"]
        except Exception:
            score = 0.0
        bleu_scores.append(score)

    pred_emb = embedder.encode(predictions, convert_to_numpy=True)
    ref_emb = embedder.encode(references, convert_to_numpy=True)

    cosine_scores = []
    for i in range(len(data)):
        cosine_scores.append(float(cosine_similarity([pred_emb[i]], [ref_emb[i]])[0][0]))

    by_domain = defaultdict(list)
    for item, bleu_s, cos_s in zip(data, bleu_scores, cosine_scores):
        by_domain[item["domain"]].append({
            "bleu": bleu_s,
            "cosine": cos_s,
        })

    domain_report = {}
    for domain, rows in by_domain.items():
        domain_report[domain] = {
            "count": len(rows),
            "avg_bleu": mean([x["bleu"] for x in rows]),
            "avg_cosine": mean([x["cosine"] for x in rows]),
        }

    report = {
        "num_samples": len(data),
        "rouge": rouge_scores,
        "avg_bleu": mean(bleu_scores),
        "avg_cosine": mean(cosine_scores),
        "by_domain": domain_report,
    }

    report_path = Path(args.report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
