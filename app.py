import streamlit as st
import torch
from threading import Thread
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from peft import PeftModel

st.set_page_config(
    page_title="Medical Chatbot Demo",
    page_icon="🩺",
    layout="wide",
)

SYSTEM_PROMPT = (
    "You are a medical knowledge chatbot for academic coursework. "
    "Provide clear, concise, educational medical answers. "
    "Do not provide diagnosis or treatment decisions."
)

MODEL_OPTIONS = {
    "Base 0.5B": {
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "adapter_path": None,
        "load_in_4bit": False,
        "description": "Original Qwen2.5-0.5B-Instruct",
    },
    "Fine-tuned 0.5B": {
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "adapter_path": "outputs/qwen25_medchat_smoke",
        "load_in_4bit": False,
        "description": "LoRA fine-tuned medical chatbot",
    },
    "Base 3B": {
        "base_model": "Qwen/Qwen2.5-3B-Instruct",
        "adapter_path": None,
        "load_in_4bit": True,
        "description": "Original Qwen2.5-3B-Instruct (4-bit)",
    },
    "Fine-tuned 3B": {
        "base_model": "Qwen/Qwen2.5-3B-Instruct",
        "adapter_path": "outputs/qwen25_medchat_3b_qlora_smoke",
        "load_in_4bit": True,
        "description": "QLoRA fine-tuned medical chatbot (4-bit)",
    },
}

EXAMPLE_QUESTIONS = [
    "What are the common symptoms of diabetes?",
    "What is hypertension?",
    "What is the difference between CT and MRI?",
    "What are common risk factors for heart disease?",
    "What is anemia?",
]

def get_quant_config(load_in_4bit: bool):
    if not load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

@st.cache_resource(show_spinner=True)
def load_model_bundle(label: str):
    cfg = MODEL_OPTIONS[label]

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
    return tokenizer, model

def stream_generate_response(tokenizer, model, query, max_new_tokens=256, temperature=0.3, top_p=0.85):
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

    partial_text = ""
    for new_text in streamer:
        partial_text += new_text
        yield partial_text

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_example" not in st.session_state:
        st.session_state.selected_example = EXAMPLE_QUESTIONS[0]

init_session_state()

# ---- Sidebar ----
with st.sidebar:
    st.title("⚙️ Settings")

    selected_model = st.selectbox(
        "Choose model",
        list(MODEL_OPTIONS.keys()),
        index=3,
    )

    st.caption(MODEL_OPTIONS[selected_model]["description"])

    max_new_tokens = st.slider(
        "Max new tokens",
        min_value=128,
        max_value=512,
        value=256,
        step=32,
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
    )

    top_p = st.slider(
        "Top-p",
        min_value=0.5,
        max_value=1.0,
        value=0.85,
        step=0.05,
    )

    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("### Project Info")
    st.markdown(
        "- **Task**: Medical-domain chatbot\n"
        "- **Datasets**: MedQuAD, PubMedQA, MedMCQA\n"
        "- **Methods**: LoRA / QLoRA\n"
        "- **Purpose**: Coursework demo"
    )

# ---- Main Header ----
st.title("🩺 Medical Chatbot Demo")
st.caption("Qwen2.5 medical-domain chatbot fine-tuning coursework demo")

st.warning(
    "This system is for academic demonstration only. "
    "It does not provide medical diagnosis, treatment, or professional clinical advice."
)

# ---- Example Questions ----
st.subheader("Example Questions")
cols = st.columns(len(EXAMPLE_QUESTIONS))
for i, q in enumerate(EXAMPLE_QUESTIONS):
    if cols[i].button(f"Example {i+1}", use_container_width=True):
        st.session_state.selected_example = q

query = st.text_area(
    "Enter your medical question",
    value=st.session_state.selected_example,
    height=120,
    placeholder="Ask a medical knowledge question...",
)

# ---- Chat History ----
st.subheader("Conversation")
chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ---- Action Buttons ----
col1, col2 = st.columns([1, 1])
generate_clicked = col1.button("Generate Response", type="primary", use_container_width=True)
continue_clicked = col2.button("Continue Generation", use_container_width=True)

# ---- Generate ----
if generate_clicked:
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        st.session_state.messages.append({"role": "user", "content": query})

        with st.spinner(f"Loading {selected_model}..."):
            tokenizer, model = load_model_bundle(selected_model)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            final_response = ""

            for partial_answer in stream_generate_response(
                tokenizer=tokenizer,
                model=model,
                query=query,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            ):
                final_response = partial_answer
                response_placeholder.markdown(final_response + "▌")

            response_placeholder.markdown(final_response)

        st.session_state.messages.append({"role": "assistant", "content": final_response})

# ---- Continue ----
if continue_clicked:
    if not st.session_state.messages or st.session_state.messages[-1]["role"] != "assistant":
        st.warning("No previous assistant response found to continue.")
    else:
        previous_answer = st.session_state.messages[-1]["content"]
        last_user_query = ""
        for msg in reversed(st.session_state.messages):
            if msg["role"] == "user":
                last_user_query = msg["content"]
                break

        continue_query = (
            f"{last_user_query}\n\n"
            f"The previous answer was cut off. Please continue from where you stopped:\n\n"
            f"{previous_answer}"
        )

        with st.spinner(f"Loading {selected_model}..."):
            tokenizer, model = load_model_bundle(selected_model)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            appended_response = ""

            for partial_answer in stream_generate_response(
                tokenizer=tokenizer,
                model=model,
                query=continue_query,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            ):
                appended_response = partial_answer
                response_placeholder.markdown(previous_answer + appended_response + "▌")

            full_response = previous_answer + appended_response
            response_placeholder.markdown(full_response)

        st.session_state.messages[-1]["content"] = full_response
