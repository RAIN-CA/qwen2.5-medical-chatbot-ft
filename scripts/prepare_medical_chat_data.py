from pathlib import Path
import json
import random

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

SYSTEM_PROMPT = (
    "You are a medical knowledge chatbot for academic coursework. "
    "Provide clear, concise, educational medical answers based on the given question. "
    "Do not claim to provide diagnosis or treatment. "
    "When appropriate, mention uncertainty and encourage consulting healthcare professionals."
)

def read_jsonl(path: Path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

def write_jsonl(items, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def make_record(dataset_name, task_type, user_text, assistant_text, metadata=None):
    return {
        "dataset": dataset_name,
        "task_type": task_type,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
        "metadata": metadata or {}
    }

def convert_medquad(split="train", max_samples=None):
    path = RAW_DIR / "medquad" / split / f"{split}.jsonl"
    rows = read_jsonl(path)
    out = []

    if max_samples is not None:
        rows = rows[:max_samples]

    for row in rows:
        question = str(row.get("Question", "")).strip()
        answer = str(row.get("Answer", "")).strip()
        qtype = str(row.get("qtype", "")).strip()

        if not question or not answer:
            continue

        user_text = question
        assistant_text = answer

        out.append(
            make_record(
                dataset_name="medquad",
                task_type="medical_qa",
                user_text=user_text,
                assistant_text=assistant_text,
                metadata={"qtype": qtype, "source_split": split},
            )
        )
    return out

def convert_pubmedqa(split="train", max_samples=None):
    path = RAW_DIR / "pubmedqa" / split / f"{split}.jsonl"
    rows = read_jsonl(path)
    out = []

    if max_samples is not None:
        rows = rows[:max_samples]

    for row in rows:
        question = str(row.get("QUESTION", "")).strip()
        contexts = row.get("CONTEXTS", [])
        final_decision = str(row.get("final_decision", "")).strip()
        long_answer = str(row.get("LONG_ANSWER", "")).strip()
        year = row.get("YEAR", None)
        meshes = row.get("MESHES", [])

        if not question:
            continue

        context_text = ""
        if isinstance(contexts, list) and contexts:
            context_text = "\n".join([str(c).strip() for c in contexts if str(c).strip()])

        user_text = question
        if context_text:
            user_text = f"Question: {question}\n\nContext:\n{context_text}"

        answer_parts = []
        if final_decision:
            answer_parts.append(f"Answer: {final_decision}")
        if long_answer:
            answer_parts.append(f"Explanation: {long_answer}")

        assistant_text = "\n\n".join(answer_parts).strip()
        if not assistant_text:
            continue

        out.append(
            make_record(
                dataset_name="pubmedqa",
                task_type="biomedical_research_qa",
                user_text=user_text,
                assistant_text=assistant_text,
                metadata={
                    "year": year,
                    "meshes": meshes,
                    "source_split": split,
                },
            )
        )
    return out

def convert_medmcqa(split="train", max_samples=None):
    path = RAW_DIR / "medmcqa" / split / f"{split}.jsonl"
    rows = read_jsonl(path)
    out = []

    if max_samples is not None:
        rows = rows[:max_samples]

    option_map = {1: "A", 2: "B", 3: "C", 4: "D"}

    for row in rows:
        question = str(row.get("question", "")).strip()
        opa = str(row.get("opa", "")).strip()
        opb = str(row.get("opb", "")).strip()
        opc = str(row.get("opc", "")).strip()
        opd = str(row.get("opd", "")).strip()
        exp = str(row.get("exp", "")).strip()
        cop = row.get("cop", None)
        subject_name = str(row.get("subject_name", "")).strip()
        topic_name = str(row.get("topic_name", "")).strip()

        if not question:
            continue

        answer_letter = option_map.get(cop, "")
        answer_text = ""
        if answer_letter == "A":
            answer_text = opa
        elif answer_letter == "B":
            answer_text = opb
        elif answer_letter == "C":
            answer_text = opc
        elif answer_letter == "D":
            answer_text = opd

        if not answer_letter or not answer_text:
            continue

        user_text = (
            f"Answer the following medical multiple-choice question.\n\n"
            f"Question: {question}\n"
            f"A. {opa}\n"
            f"B. {opb}\n"
            f"C. {opc}\n"
            f"D. {opd}"
        )

        assistant_text = f"The correct answer is {answer_letter}. {answer_text}"
        if exp:
            assistant_text += f"\n\nExplanation: {exp}"

        out.append(
            make_record(
                dataset_name="medmcqa",
                task_type="medical_mcqa",
                user_text=user_text,
                assistant_text=assistant_text,
                metadata={
                    "subject_name": subject_name,
                    "topic_name": topic_name,
                    "source_split": split,
                },
            )
        )
    return out

def build_and_save():
    # 先做一个适合你机器的版本
    train_records = []
    val_records = []
    test_records = []

    # 训练集采样策略：不要一上来全量
    train_records.extend(convert_medquad("train", max_samples=12000))
    train_records.extend(convert_pubmedqa("train", max_samples=450))
    train_records.extend(convert_medmcqa("train", max_samples=20000))

    # 验证集
    val_records.extend(convert_pubmedqa("validation", max_samples=50))
    val_records.extend(convert_medmcqa("validation", max_samples=1000))

    # MedQuAD 原始没有 validation/test，这里从 train 里切一小部分作为测试参考
    medquad_all = convert_medquad("train", max_samples=16407)
    random.shuffle(medquad_all)
    medquad_val_extra = medquad_all[:500]
    medquad_test_extra = medquad_all[500:1000]

    val_records.extend(medquad_val_extra)
    test_records.extend(medquad_test_extra)

    # 测试集
    test_records.extend(convert_pubmedqa("test", max_samples=200))
    test_records.extend(convert_medmcqa("test", max_samples=1000))

    # 打乱
    random.shuffle(train_records)
    random.shuffle(val_records)
    random.shuffle(test_records)

    # 保存完整版
    write_jsonl(train_records, PROCESSED_DIR / "train.jsonl")
    write_jsonl(val_records, PROCESSED_DIR / "validation.jsonl")
    write_jsonl(test_records, PROCESSED_DIR / "test.jsonl")

    # 保存一个 smoke test 小子集
    small_train = train_records[:1000]
    small_val = val_records[:200]
    small_test = test_records[:200]

    write_jsonl(small_train, PROCESSED_DIR / "small_train.jsonl")
    write_jsonl(small_val, PROCESSED_DIR / "small_validation.jsonl")
    write_jsonl(small_test, PROCESSED_DIR / "small_test.jsonl")

    summary = {
        "train_size": len(train_records),
        "validation_size": len(val_records),
        "test_size": len(test_records),
        "small_train_size": len(small_train),
        "small_validation_size": len(small_val),
        "small_test_size": len(small_test),
    }

    with open(PROCESSED_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Saved processed datasets.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    build_and_save()
