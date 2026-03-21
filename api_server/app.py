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
)
from rag_service import retrieve_context, clear_chunk_cache

app = Flask(__name__)
CORS(app)

SYSTEM_PROMPT = (
    "You are a medical knowledge chatbot for academic coursework. "
    "Provide brief, clear, educational medical answers by default. "
    "Only give a longer or more detailed answer if the user explicitly asks for more detail. "
    "Do not provide diagnosis or treatment decisions. "
    "If retrieved context is provided, use it as supporting reference. "
    "If the retrieved context is insufficient, answer cautiously and say so."
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

MAX_UPLOAD_MB = 20
MODEL_CACHE = {}

STREAMS = {}
STREAM_LOCK = threading.Lock()


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
    stream["queue"].put({
        "type": event_type,
        "data": data,
    })


def mark_done(stream_id):
    with STREAM_LOCK:
        stream = STREAMS.get(stream_id)
    if stream:
        stream["done"] = True


def cleanup_stream(stream_id):
    with STREAM_LOCK:
        if stream_id in STREAMS:
            del STREAMS[stream_id]


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
        put_event(
            stream_id,
            "status_done",
            {"id": "load", "text": "Model ready" if loaded_now else "Model loaded from cache"},
        )

        rag_context = ""
        rag_hits = []

        if use_rag and selected_files:
            put_event(stream_id, "status", {"id": "rag", "text": "Reading and chunking documents"})
            time.sleep(0.06)
            put_event(stream_id, "status_update", {"id": "rag", "text": "Computing similarity scores"})

            rag_context, rag_hits = retrieve_context(
                query=query,
                selected_files=selected_files,
                top_k=rag_top_k,
                chunk_size=rag_chunk_size,
                overlap=rag_overlap,
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

        user_content = query
        if rag_context:
            user_content = (
                f"User question:\n{query}\n\n"
                f"Retrieved context:\n{rag_context}\n\n"
                f"Use the retrieved context if relevant, but avoid unsupported claims."
            )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        put_event(stream_id, "status_done", {"id": "prompt", "text": "Prompt ready"})
        put_event(stream_id, "status", {"id": "generate", "text": "Generating response"})

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

        worker = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        worker.start()

        final_text = ""
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
            {"key": k, "label": v["label"]}
            for k, v in MODEL_OPTIONS.items()
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

    return jsonify({
        "context": context,
        "hits": hits,
    })


@app.post("/api/chat/start")
def chat_start():
    payload = request.get_json(force=True)
    model_key = payload.get("model_key", "ft_3b")
    query = payload.get("query", "").strip()
    max_new_tokens = int(payload.get("max_new_tokens", 384))
    temperature = float(payload.get("temperature", 0.2))
    top_p = float(payload.get("top_p", 0.85))
    use_rag = bool(payload.get("use_rag", False))
    selected_files = payload.get("selected_files", []) or []
    rag_top_k = int(payload.get("rag_top_k", 4))
    rag_chunk_size = int(payload.get("rag_chunk_size", 800))
    rag_overlap = int(payload.get("rag_overlap", 120))

    if model_key not in MODEL_OPTIONS:
        return jsonify({"error": "Invalid model key"}), 400

    if not query:
        return jsonify({"error": "Empty query"}), 400

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
