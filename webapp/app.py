from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

app = Flask(__name__)

SYSTEM_PROMPT = (
    "You are a medical knowledge chatbot for academic coursework. "
    "Provide clear, concise, educational medical answers. "
    "Do not provide diagnosis or treatment decisions."
)

MODEL_OPTIONS = {
    "base_0_5b": {
        "label": "Base 0.5B",
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "adapter_path": None,
        "load_in_4bit": False,
    },
    "ft_0_5b": {
        "label": "Fine-tuned 0.5B",
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "adapter_path": "outputs/qwen25_medchat_smoke",
        "load_in_4bit": False,
    },
    "base_3b": {
        "label": "Base 3B",
        "base_model": "Qwen/Qwen2.5-3B-Instruct",
        "adapter_path": None,
        "load_in_4bit": True,
    },
    "ft_3b": {
        "label": "Fine-tuned 3B",
        "base_model": "Qwen/Qwen2.5-3B-Instruct",
        "adapter_path": "outputs/qwen25_medchat_3b_qlora_smoke",
        "load_in_4bit": True,
    },
}

MODEL_CACHE = {}

def get_quant_config(load_in_4bit: bool):
    if not load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

def load_model_bundle(model_key: str):
    if model_key in MODEL_CACHE:
        return MODEL_CACHE[model_key]

    cfg = MODEL_OPTIONS[model_key]

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["base_model"],
        trust_remote_code=True,
    )

    quant_config = get_quant_config(cfg["load_in_4bit"])

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=quant_config,
    )

    model = base_model
    if cfg["adapter_path"]:
        model = PeftModel.from_pretrained(base_model, cfg["adapter_path"])

    model.eval()
    MODEL_CACHE[model_key] = (tokenizer, model)
    return tokenizer, model

def generate_response(model_key: str, query: str, max_new_tokens: int = 256):
    tokenizer, model = load_model_bundle(model_key)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.85,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response.strip()

@app.route("/")
def index():
    return render_template("index.html", model_options=MODEL_OPTIONS)

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    model_key = data.get("model_key", "ft_3b")
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Empty query"}), 400

    try:
        answer = generate_response(model_key, query)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
