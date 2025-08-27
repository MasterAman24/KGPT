import os
from langchain_groq import ChatGroq

#DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
#DEFAULT_GROQ_MODEL= "qwen/qwen3-32b"

def make_llm(model: str = DEFAULT_GROQ_MODEL) -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set.")
    return ChatGroq(api_key=api_key, model=model)
