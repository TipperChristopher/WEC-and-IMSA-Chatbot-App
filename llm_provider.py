import os
from typing import Any

def _make_ollama(base_url: str, model: str) -> Any:
    from langchain_community.llms import Ollama
    return Ollama(model=model, base_url=base_url)


class OpenAIWrapper:
    def __init__(self, model: str):
        try:
            import openai
        except Exception as e:
            raise RuntimeError("openai package required for OpenAI provider") from e

        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            openai.api_key = api_key
        self._openai = openai
        self.model = model

    def invoke(self, prompt: str) -> str:
        # Try chat completion then fallback to completion
        try:
            resp = self._openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
            )
            return resp["choices"][0]["message"]["content"]
        except Exception:
            resp = self._openai.Completion.create(model=self.model, prompt=prompt, max_tokens=1024)
            return resp["choices"][0]["text"]


def get_llm():
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    if provider == "ollama":
        base_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
        return _make_ollama(base_url=base_url, model=model)
    elif provider == "openai":
        model = os.getenv("OPENAI_MODEL", "gpt-4o")
        return OpenAIWrapper(model=model)
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
