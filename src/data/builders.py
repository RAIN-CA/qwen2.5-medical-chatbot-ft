import json
import random
from pathlib import Path

from src.data.io import write_jsonl
from src.data.medical.medquad import convert_medquad
from src.data.medical.pubmedqa import convert_pubmedqa
from src.data.medical.medmcqa import convert_medmcqa

from src.data.finance.finance_qa import convert_fiqa, convert_financial_qa_10k
from src.data.legal.legal_qa import convert_legalqaeval, convert_australian_legal
from src.data.general.general_qa import convert_squad


def build_medical_datasets(
    raw_dir: Path,
    processed_dir: Path,
    random_seed: int = 42,
):
    random.seed(random_seed)
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_records = []
    val_records = []
    test_records = []

    train_records.extend(convert_medquad(raw_dir, "train", max_samples=12000))
    train_records.extend(convert_pubmedqa(raw_dir, "train", max_samples=450))
    train_records.extend(convert_medmcqa(raw_dir, "train", max_samples=20000))

    val_records.extend(convert_pubmedqa(raw_dir, "validation", max_samples=50))
    val_records.extend(convert_medmcqa(raw_dir, "validation", max_samples=1000))

    medquad_all = convert_medquad(raw_dir, "train", max_samples=16407)
    random.shuffle(medquad_all)
    medquad_val_extra = medquad_all[:500]
    medquad_test_extra = medquad_all[500:1000]

    val_records.extend(medquad_val_extra)
    test_records.extend(medquad_test_extra)

    test_records.extend(convert_pubmedqa(raw_dir, "test", max_samples=200))
    test_records.extend(convert_medmcqa(raw_dir, "test", max_samples=1000))

    random.shuffle(train_records)
    random.shuffle(val_records)
    random.shuffle(test_records)

    write_jsonl(train_records, processed_dir / "train.jsonl")
    write_jsonl(val_records, processed_dir / "validation.jsonl")
    write_jsonl(test_records, processed_dir / "test.jsonl")

    small_train = train_records[:1000]
    small_val = val_records[:200]
    small_test = test_records[:200]

    write_jsonl(small_train, processed_dir / "small_train.jsonl")
    write_jsonl(small_val, processed_dir / "small_validation.jsonl")
    write_jsonl(small_test, processed_dir / "small_test.jsonl")

    summary = {
        "train_size": len(train_records),
        "validation_size": len(val_records),
        "test_size": len(test_records),
        "small_train_size": len(small_train),
        "small_validation_size": len(small_val),
        "small_test_size": len(small_test),
    }

    with open(processed_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def build_general_datasets(
    processed_dir: Path,
    random_seed: int = 42,
):
    random.seed(random_seed)

    general = convert_squad("data/raw/general/squad")

    write_jsonl(general, processed_dir / "general.jsonl")
    write_jsonl(general[:2000], processed_dir / "general_small.jsonl")

    return {
        "general": len(general),
        "general_small": min(2000, len(general)),
    }


def build_multidomain_datasets(
    processed_dir: Path,
    random_seed: int = 42,
):
    random.seed(random_seed)
    processed_dir.mkdir(parents=True, exist_ok=True)

    finance = (
        convert_fiqa("data/raw/finance/fiqa_main") +
        convert_financial_qa_10k("data/raw/finance/financial_qa_10k")
    )

    legal = (
        convert_legalqaeval("data/raw/legal/legalqaeval") +
        convert_australian_legal("data/raw/legal/open_australian_legal_qa")
    )

    general = convert_squad("data/raw/general/squad")

    random.shuffle(finance)
    random.shuffle(legal)
    random.shuffle(general)

    write_jsonl(finance, processed_dir / "finance.jsonl")
    write_jsonl(legal, processed_dir / "legal.jsonl")
    write_jsonl(general, processed_dir / "general.jsonl")

    write_jsonl(finance[:2000], processed_dir / "finance_small.jsonl")
    write_jsonl(legal[:2000], processed_dir / "legal_small.jsonl")
    write_jsonl(general[:2000], processed_dir / "general_small.jsonl")

    summary = {
        "finance": len(finance),
        "finance_small": min(2000, len(finance)),
        "legal": len(legal),
        "legal_small": min(2000, len(legal)),
        "general": len(general),
        "general_small": min(2000, len(general)),
    }

    with open(processed_dir / "multidomain_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary
