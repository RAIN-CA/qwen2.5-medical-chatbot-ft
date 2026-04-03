from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from werkzeug.utils import secure_filename
import json
import os
import time
import uuid
import queue
import threading

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from peft import PeftModel

from document_store import (
    UPLOAD_DIR,
    ALLOWED_EXTENSIONS,
    get_file_metadata_list,
    delete_file,
    clear_file_cache,
    reset_document_runtime_cache,
)
from rag_service import retrieve_context, clear_chunk_cache, reset_rag_runtime_cache

app = Flask(__name__)
CORS(app)

SYSTEM_PROMPT = (
    "You are a medical knowledge chatbot for academic coursework. "
    "Answer briefly and clearly by default. "
    "Keep most answers within 3 to 6 sentences unless the user explicitly asks for more detail. "
    "Avoid repetition, avoid multi-turn roleplay, and stop once the answer is complete. "
    "Do not provide diagnosis or treatment decisions. "
    "If retrieved context is provided, use it as supporting reference. "
    "If the retrieved context is insufficient, answer cautiously and say so."
)

MAX_UPLOAD_MB = 20
MODEL_CACHE = {}
LAST_MODEL_RUNTIME_CONFIG = {}

STREAMS = {}
STREAM_LOCK = threading.Lock()


def initialize_rag_runtime():
    reset_document_runtime_cache()
    reset_rag_runtime_cache()


initialize_rag_runtime()


def get_quant_config(load_in_4bit: bool):
    if not load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def infer_model_domain(variant: str):
    lower = variant.lower()
    if "balanced_multidomain" in lower:
        return "multidomain"
    if "medical" in lower:
        return "medical"
    if "finance" in lower:
        return "finance"
    if "legal" in lower:
        return "legal"
    if "general" in lower:
        return "general"
    if "base" in lower:
        return "all"
    return "all"


def scan_experiment_models():
    root = "outputs/experiments"
    if not os.path.exists(root):
        return {}

    models = {}

    for family in sorted(os.listdir(root)):
        family_path = os.path.join(root, family)
        if not os.path.isdir(family_path):
            continue

        for size in sorted(os.listdir(family_path)):
            size_path = os.path.join(family_path, size)
            if not os.path.isdir(size_path):
                continue

            for variant in sorted(os.listdir(size_path)):
                variant_path = os.path.join(size_path, variant)
                adapter_path = os.path.join(variant_path, "adapter")

                if not os.path.isdir(adapter_path):
                    continue

                key = f"{family}_{size}_{variant}"
                domain = infer_model_domain(variant)

                if family == "qwen25":
                    base_model = (
                        "Qwen/Qwen2.5-3B-Instruct"
                        if "3b" in size
                        else "Qwen/Qwen2.5-0.5B-Instruct"
                    )
                    model_type = "causal"
                    load_in_4bit = "3b" in size
                    prompt_style = "chat"
                elif family == "bart":
                    base_model = "facebook/bart-base"
                    model_type = "seq2seq"
                    load_in_4bit = False
                    prompt_style = "plain"
                elif family == "t5":
                    base_model = "t5-small"
                    model_type = "seq2seq"
                    load_in_4bit = False
                    prompt_style = "t5"
                else:
                    continue

                models[key] = {
                    "key": key,
                    "label": f"{family} / {size} / {variant}",
                    "family": family,
                    "size": size,
                    "variant": variant,
                    "domain": domain,
                    "base_model": base_model,
                    "adapter_path": adapter_path,
                    "load_in_4bit": load_in_4bit,
                    "model_type": model_type,
                    "prompt_style": prompt_style,
                }

    # fallback legacy smoke models if experiments dir is unavailable
    if not models:
        models = {
            "base_0_5b": {
                "key": "base_0_5b",
                "label": "Base 0.5B",
                "family": "qwen25",
                "size": "0.5b",
                "variant": "base",
                "domain": "all",
                "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
                "adapter_path": None,
                "load_in_4bit": False,
                "model_type": "causal",
                "prompt_style": "chat",
            },
            "ft_0_5b": {
                "key": "ft_0_5b",
                "label": "Fine-tuned 0.5B",
                "family": "qwen25",
                "size": "0.5b",
                "variant": "medical_ft",
                "domain": "medical",
                "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
                "adapter_path": "outputs/qwen25_medchat_smoke",
                "load_in_4bit": False,
                "model_type": "causal",
                "prompt_style": "chat",
            },
            "base_3b": {
                "key": "base_3b",
                "label": "Base 3B",
                "family": "qwen25",
                "size": "3b",
                "variant": "base",
                "domain": "all",
                "base_model": "Qwen/Qwen2.5-3B-Instruct",
                "adapter_path": None,
                "load_in_4bit": True,
                "model_type": "causal",
                "prompt_style": "chat",
            },
            "ft_3b": {
                "key": "ft_3b",
                "label": "Fine-tuned 3B",
                "family": "qwen25",
                "size": "3b",
                "variant": "medical_ft",
                "domain": "medical",
                "base_model": "Qwen/Qwen2.5-3B-Instruct",
                "adapter_path": "outputs/qwen25_medchat_3b_qlora_smoke",
                "load_in_4bit": True,
                "model_type": "causal",
                "prompt_style": "chat",
            },
        }

    return models


MODEL_OPTIONS = scan_experiment_models()


def build_user_content(query: str, rag_context: str):
    if not rag_context:
        return query
    return (
        f"User question:\n{query}\n\n"
        f"Retrieved context:\n{rag_context}\n\n"
        f"Use the retrieved context if relevant, but avoid unsupported claims."
    )


def build_input_text(tokenizer, cfg, query: str, rag_context: str):
    prompt_style = cfg.get("prompt_style", "chat")
    user_content = build_user_content(query, rag_context)

    if prompt_style == "chat":
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    if prompt_style == "t5":
        return f"question: {user_content}"

    return user_content



def maybe_reload_model_for_runtime_change(model_key: str, max_new_tokens: int, temperature: float, top_p: float):
    runtime_sig = {
        "max_new_tokens": int(max_new_tokens),
        "temperature": round(float(temperature), 4),
        "top_p": round(float(top_p), 4),
    }

    prev_sig = LAST_MODEL_RUNTIME_CONFIG.get(model_key)
    if prev_sig is not None and prev_sig != runtime_sig:
        if model_key in MODEL_CACHE:
            del MODEL_CACHE[model_key]

    LAST_MODEL_RUNTIME_CONFIG[model_key] = runtime_sig


def load_model_bundle(model_key: str):
    if model_key in MODEL_CACHE:
        return MODEL_CACHE[model_key], False

    cfg = MODEL_OPTIONS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["base_model"],
        trust_remote_code=True,
    )

    quant_config = get_quant_config(cfg["load_in_4bit"])

    if cfg["model_type"] == "causal":
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
    else:
        # seq2seq experiments were saved as full fine-tuned models in adapter dir
        model = AutoModelForSeq2SeqLM.from_pretrained(
            cfg["adapter_path"] or cfg["base_model"],
            dtype=torch.float16,
            device_map="auto",
        )

    model.eval()
    MODEL_CACHE[model_key] = (tokenizer, model)
    return (tokenizer, model), True


def create_stream():
    stream_id = str(uuid.uuid4())
    with STREAM_LOCK:
        STREAMS[stream_id] = {
            "queue": queue.Queue(),
            "done": False,
            "created_at": time.time(),
        }
    return stream_id


def put_event(stream_id, event_type, data):
    with STREAM_LOCK:
        stream = STREAMS.get(stream_id)
    if not stream:
        return
    stream["queue"].put({"type": event_type, "data": data})


def mark_done(stream_id):
    with STREAM_LOCK:
        stream = STREAMS.get(stream_id)
    if stream:
        stream["done"] = True


def cleanup_stream(stream_id):
    with STREAM_LOCK:
        if stream_id in STREAMS:
            del STREAMS[stream_id]


def stream_text_manually(stream_id, text: str):
    emitted_chars = 0
    for ch in text:
        emitted_chars += 1
        put_event(stream_id, "chunk", {"text": ch})

        if emitted_chars % 40 == 0:
            put_event(
                stream_id,
                "status_update",
                {"id": "generate", "text": f"Generating response ({emitted_chars} chars)"},
            )

        time.sleep(0.01)

    return emitted_chars


def background_generate(
    stream_id,
    model_key,
    query,
    max_new_tokens,
    temperature,
    top_p,
    use_rag,
    selected_files,
    rag_top_k,
    rag_chunk_size,
    rag_overlap,
):
    try:
        put_event(stream_id, "status", {"id": "load", "text": "Checking model cache"})
        (tokenizer, model), loaded_now = load_model_bundle(model_key)
        cfg = MODEL_OPTIONS[model_key]
        put_event(
            stream_id,
            "status_done",
            {"id": "load", "text": "Model ready" if loaded_now else "Model loaded from cache"},
        )

        rag_context = ""
        rag_hits = []

        if use_rag and selected_files:
            rag_context, rag_hits = retrieve_context(
                query=query,
                selected_files=selected_files,
                top_k=rag_top_k,
                chunk_size=rag_chunk_size,
                overlap=rag_overlap,
                progress_callback=lambda event_type, step_id, msg: put_event(
                    stream_id,
                    event_type,
                    {"id": step_id, "text": msg},
                ),
            )

            if rag_context:
                put_event(
                    stream_id,
                    "status_done",
                    {
                        "id": "rag",
                        "text": f"Retrieved {len(rag_hits)} relevant chunks from {len(selected_files)} file(s)",
                    },
                )
            else:
                put_event(
                    stream_id,
                    "status_done",
                    {"id": "rag", "text": "No strongly relevant chunks found"},
                )

        put_event(stream_id, "status", {"id": "prompt", "text": "Building prompt"})
        text = build_input_text(tokenizer, cfg, query, rag_context)
        inputs = tokenizer(text, return_tensors="pt", truncation=True).to(model.device)
        put_event(stream_id, "status_done", {"id": "prompt", "text": "Prompt ready"})
        put_event(stream_id, "status", {"id": "generate", "text": "Generating response"})

        final_text = ""

        if cfg["model_type"] == "causal":
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
                repetition_penalty=1.08,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            worker = threading.Thread(target=model.generate, kwargs=generation_kwargs)
            worker.start()

            emitted_chars = 0
            for new_text in streamer:
                for ch in new_text:
                    final_text += ch
                    emitted_chars += 1
                    put_event(stream_id, "chunk", {"text": ch})

                    if emitted_chars % 40 == 0:
                        put_event(
                            stream_id,
                            "status_update",
                            {"id": "generate", "text": f"Generating response ({emitted_chars} chars)"},
                        )

                    time.sleep(0.012)
        else:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    repetition_penalty=1.05,
                )
            final_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            emitted_chars = stream_text_manually(stream_id, final_text)

        put_event(
            stream_id,
            "status_done",
            {"id": "generate", "text": f"Generation complete ({emitted_chars} chars)"},
        )
        put_event(stream_id, "done", {"text": final_text, "rag_hits": rag_hits})
        mark_done(stream_id)

    except Exception as e:
        put_event(stream_id, "error", {"message": str(e)})
        mark_done(stream_id)


@app.get("/api/models")
def get_models():
    return jsonify({
        "models": [
            {
                "key": k,
                "label": v["label"],
                "family": v["family"],
                "size": v["size"],
                "variant": v["variant"],
                "domain": v["domain"],
                "model_type": v["model_type"],
            }
            for k, v in sorted(MODEL_OPTIONS.items(), key=lambda x: x[1]["label"])
        ]
    })


@app.get("/api/rag/files")
def get_rag_files():
    return jsonify({"files": get_file_metadata_list()})


@app.post("/api/rag/files/upload")
def upload_rag_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    f = request.files["file"]
    if not f or not f.filename:
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(f.filename)
    suffix = os.path.splitext(filename)[1].lower()

    if suffix not in ALLOWED_EXTENSIONS:
        return jsonify({
            "error": f"Unsupported file type: {suffix}. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        }), 400

    f.seek(0, os.SEEK_END)
    size_bytes = f.tell()
    f.seek(0)

    if size_bytes > MAX_UPLOAD_MB * 1024 * 1024:
        return jsonify({"error": f"File too large. Max size is {MAX_UPLOAD_MB} MB"}), 400

    save_path = UPLOAD_DIR / filename
    f.save(str(save_path))
    clear_file_cache(filename)
    clear_chunk_cache(filename)

    return jsonify({
        "message": "Upload successful",
        "filename": filename,
        "files": get_file_metadata_list(),
    })


@app.delete("/api/rag/files/<path:filename>")
def delete_rag_file(filename):
    deleted = delete_file(filename)
    clear_chunk_cache(filename)

    if not deleted:
        return jsonify({"error": "File not found"}), 404

    return jsonify({
        "message": "File deleted",
        "filename": filename,
        "files": get_file_metadata_list(),
    })


@app.post("/api/rag/retrieve")
def rag_retrieve():
    payload = request.get_json(force=True)
    query = payload.get("query", "").strip()
    selected_files = payload.get("selected_files", []) or []
    top_k = int(payload.get("top_k", 4))
    chunk_size = int(payload.get("chunk_size", 800))
    overlap = int(payload.get("overlap", 120))

    if not query:
        return jsonify({"error": "Empty query"}), 400

    context, hits = retrieve_context(
        query=query,
        selected_files=selected_files,
        top_k=top_k,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    return jsonify({"context": context, "hits": hits})


@app.post("/api/rag/reset")
def reset_rag_runtime_endpoint():
    reset_document_runtime_cache()
    reset_rag_runtime_cache()
    return jsonify({"message": "RAG runtime cache reset successfully"})


@app.post("/api/chat/start")
def chat_start():
    payload = request.get_json(force=True)
    default_model_key = next(iter(MODEL_OPTIONS.keys()), "")
    model_key = payload.get("model_key", default_model_key)
    query = payload.get("query", "").strip()
    max_new_tokens = int(payload.get("max_new_tokens", 160))
    temperature = float(payload.get("temperature", 0.8))
    top_p = float(payload.get("top_p", 0.5))
    use_rag = bool(payload.get("use_rag", False))
    selected_files = payload.get("selected_files", []) or []
    rag_top_k = int(payload.get("rag_top_k", 4))
    rag_chunk_size = int(payload.get("rag_chunk_size", 800))
    rag_overlap = int(payload.get("rag_overlap", 120))

    if model_key not in MODEL_OPTIONS:
        return jsonify({"error": "Invalid model key"}), 400

    if not query:
        return jsonify({"error": "Empty query"}), 400

    maybe_reload_model_for_runtime_change(
        model_key=model_key,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    stream_id = create_stream()

    thread = threading.Thread(
        target=background_generate,
        args=(
            stream_id,
            model_key,
            query,
            max_new_tokens,
            temperature,
            top_p,
            use_rag,
            selected_files,
            rag_top_k,
            rag_chunk_size,
            rag_overlap,
        ),
        daemon=True,
    )
    thread.start()

    return jsonify({"stream_id": stream_id})


@app.get("/api/chat/events/<stream_id>")
def chat_events(stream_id):
    @stream_with_context
    def event_stream():
        while True:
            with STREAM_LOCK:
                stream = STREAMS.get(stream_id)

            if not stream:
                break

            try:
                item = stream["queue"].get(timeout=0.5)
                payload = json.dumps(item, ensure_ascii=False)
                yield f"data: {payload}\n\n"
            except queue.Empty:
                if stream["done"]:
                    break
                yield ": keep-alive\n\n"

        cleanup_stream(stream_id)

    response = Response(event_stream(), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    response.headers["Connection"] = "keep-alive"
    return response


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True, use_reloader=False)
