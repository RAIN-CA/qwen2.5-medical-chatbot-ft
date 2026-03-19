# qwen2.5-medical-chatbot-ft
Fine-tuning Qwen2.5 for a medical-domain chatbot using MedQuAD, PubMedQA, and MedMCQA with LoRA/QLoRA.

# qwen2.5-medical-chatbot-ft

Fine-tuning Qwen2.5 for a medical-domain chatbot using MedQuAD, PubMedQA, and MedMCQA with LoRA/QLoRA.

## 1. Project Overview

This course project aims to fine-tune a Qwen2.5 large language model (LLM, not VLM) into a **medical-domain knowledge chatbot**.

The chatbot is intended for:
- medical knowledge question answering,
- symptom and disease-related educational dialogue,
- explanation of medical tests or concepts,
- academic experimentation in domain-specific LLM adaptation.

> Disclaimer: This project is for academic research and coursework only. It is **not** intended for real-world diagnosis, treatment recommendation, or clinical deployment.

---

## 2. Target Datasets

This project uses three medical datasets with complementary roles:

1. **MedQuAD / MedQuad**
   - Medical question-answering dataset
   - Used as the main conversational medical QA source

2. **PubMedQA**
   - Biomedical research QA dataset
   - Used to improve scientific/abstract-based medical answering ability

3. **MedMCQA**
   - Large-scale medical multiple-choice QA dataset
   - Converted into instruction/chat style data for broader medical knowledge coverage

---

## 3. Project Goals

- Build a domain-specific medical chatbot based on Qwen2.5
- Explore parameter-efficient fine-tuning using LoRA / QLoRA
- Unify heterogeneous medical datasets into one instruction/chat format
- Evaluate model quality on medical QA tasks

---

## 4. Recommended Model Scope

Given limited hardware resources (e.g. RTX 5060 8GB VRAM + 16GB RAM), recommended starting points are:

- `Qwen/Qwen2.5-0.5B-Instruct`
- `Qwen/Qwen2.5-1.5B-Instruct` (only if memory allows)

Recommended first experiment:
- Start with **0.5B + LoRA/QLoRA**
- Use a subset of all three datasets
- Run a full end-to-end pipeline before scaling up

---

## 5. Project Structure

```text
qwen2.5-medical-chatbot-ft/
├── README.md
├── .gitignore
├── environment.yml
├── requirements.txt
├── configs/
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── notebooks/
├── scripts/
├── src/
│   ├── data/
│   ├── train/
│   ├── eval/
│   ├── inference/
│   └── utils/
├── outputs/
└── reports/
```
