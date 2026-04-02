DEFAULT_SYSTEM_PROMPTS = {
    "medical": (
        "You are a medical knowledge chatbot for academic coursework. "
        "Provide clear, concise, educational medical answers. "
        "Do not provide diagnosis or treatment decisions."
    ),
    "general": (
        "You are a helpful academic chatbot. "
        "Provide clear, concise, and accurate educational answers."
    ),
}


def get_system_prompt(domain: str | None = None, override_prompt: str | None = None) -> str:
    if override_prompt:
        return override_prompt

    if domain is None:
        domain = "medical"

    return DEFAULT_SYSTEM_PROMPTS.get(domain, DEFAULT_SYSTEM_PROMPTS["general"])


def build_messages(system_prompt: str, user_query: str):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]
