from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import json
import torch
from threading import Thread
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from peft import PeftModel

app = Flask(__name__)
CORS(app)

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


def emit_event(event_type, data):
    return json.dumps({"type": event_type, "data": data}, ensure_ascii=False) + "\n"


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
        return MODEL_CACHE[model_key], False

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
    return (tokenizer, model), True


@app.get("/api/models")
def get_models():
    return jsonify({
        "models": [
            {"key": k, "label": v["label"]}
            for k, v in MODEL_OPTIONS.items()
        ]
    })


@app.post("/api/chat/stream")
def chat_stream():
    payload = request.get_json(force=True)
    model_key = payload.get("model_key", "ft_3b")
    query = payload.get("query", "").strip()
    max_new_tokens = int(payload.get("max_new_tokens", 256))
    temperature = float(payload.get("temperature", 0.3))
    top_p = float(payload.get("top_p", 0.85))

    if model_key not in MODEL_OPTIONS:
        return jsonify({"error": "Invalid model key"}), 400

    if not query:
        return jsonify({"error": "Empty query"}), 400

    @stream_with_context
    def generate():
        try:
            yield emit_event("status", {
                "id": "load",
                "text": "Checking model cache"
            })

            (tokenizer, model), loaded_now = load_model_bundle(model_key)

            yield emit_event("status_done", {
                "id": "load",
                "text": "Model ready" if loaded_now else "Model loaded from cache"
            })

            yield emit_event("status", {
                "id": "prompt",
                "text": "Building prompt"
            })

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

            yield emit_event("status_done", {
                "id": "prompt",
                "text": "Prompt ready"
            })

            yield emit_event("status", {
                "id": "generate",
                "text": "Generating response"
            })

            streamer = TextIteratorStreamer(
                tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            final_text = ""
            for new_text in streamer:
                final_text += new_text
                yield emit_event("chunk", {"text": new_text})

            yield emit_event("status_done", {
                "id": "generate",
                "text": "Generation complete"
            })
            yield emit_event("done", {"text": final_text})

        except Exception as e:
            yield emit_event("error", {"message": str(e)})

    return Response(generate(), mimetype="application/x-ndjson")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True)
