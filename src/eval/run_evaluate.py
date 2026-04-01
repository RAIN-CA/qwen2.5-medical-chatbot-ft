import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


EVAL_MODELS = {
    "bert_base": "bert-base-uncased",
    "biobert": "dmis-lab/biobert-v1.1",  
    "xlnet": "xlnet-base-cased",         
    "e5": "intfloat/e5-base-v2"          
}

GROUND_TRUTH = {
    "What are the common symptoms of diabetes?": "Common symptoms include increased thirst, frequent urination, fatigue, blurred vision, and unexplained weight loss.",
    "What is hypertension?": "Hypertension is high blood pressure, a chronic condition where blood pressure in arteries is persistently elevated.",
    "What is the difference between CT and MRI?": "CT uses X-rays for fast bone/lung imaging; MRI uses magnetic fields for soft tissue, brain, and spinal cord with no radiation.",
    "What are common risk factors for heart disease?": "Risk factors include high blood pressure, high cholesterol, smoking, diabetes, obesity, and family history.",
    "What is anemia?": "Anemia is a condition defined by a lack of healthy red blood cells, leading to reduced oxygen flow to body tissues."
}

def load_evaluators():
    tokenizers = {}
    models = {}
    for name, path in EVAL_MODELS.items():
        tokenizers[name] = AutoTokenizer.from_pretrained(path)
        models[name] = AutoModel.from_pretrained(path).eval()
    return tokenizers, models


def get_score(ans1, ans2, tokenizer, model):
    inputs1 = tokenizer(ans1, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs2 = tokenizer(ans2, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        e1 = model(**inputs1).last_hidden_state[:,0,:].numpy()
        e2 = model(**inputs2).last_hidden_state[:,0,:].numpy()
    return float(cosine_similarity(e1, e2)[0][0])

def evaluate_from_json(json_path="outputs/model_comparison_results.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tokenizers, models = load_evaluators()
    all_results = []

    for model_out in data:
        model_name = model_out["model_name"]
        print(f"\nEvaluating {model_name}...")

        for item in model_out["answers"]:
            q = item["question"]
            gen_ans = item["answer"]
            gt_ans = GROUND_TRUTH[q]

            scores = {}
            for m_name in EVAL_MODELS:
                scores[m_name] = round(get_score(gen_ans, gt_ans, tokenizers[m_name], models[m_name]), 4)

            all_results.append({
                "model": model_name,
                "question": q,
                "generated": gen_ans,
                "ground_truth": gt_ans,
                **scores
            })

   
    df = pd.DataFrame(all_results)
    df.to_csv("outputs/4model_evaluation_scores.csv", index=False, encoding="utf-8-sig")
    print("\nEvaluation finished. Results saved to outputs/4model_evaluation_scores.csv")

if __name__ == "__main__":
    evaluate_from_json()