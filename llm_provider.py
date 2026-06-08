import os
import time
import random
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
        def _call():
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

        return _perform_with_retries(_call)


def _perform_with_retries(func, max_attempts: int = 3, base_delay: float = 0.5):
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            last_exc = e
            if attempt == max_attempts:
                raise
            sleep_for = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            time.sleep(sleep_for)
    raise last_exc


class AzureOpenAIWrapper:
    def __init__(self, deployment: str):
        try:
            import openai
        except Exception as e:
            raise RuntimeError("openai package required for Azure OpenAI provider") from e

        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        key = os.getenv("AZURE_OPENAI_KEY")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-10-01")
        if not endpoint or not key:
            raise RuntimeError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY must be set for Azure provider")

        openai.api_type = "azure"
        openai.api_base = endpoint.rstrip("/")
        openai.api_key = key
        openai.api_version = api_version
        self._openai = openai
        self.deployment = deployment

    def invoke(self, prompt: str) -> str:
        def _call():
            resp = self._openai.ChatCompletion.create(
                engine=self.deployment,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
            )
            return resp["choices"][0]["message"]["content"]

        return _perform_with_retries(_call)


def get_llm():
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    if provider == "ollama":
        base_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
        return _make_ollama(base_url=base_url, model=model)
    elif provider == "openai":
        model = os.getenv("OPENAI_MODEL", "gpt-4o")
        return OpenAIWrapper(model=model)
    elif provider == "azure":
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("AZURE_OPENAI_MODEL")
        if not deployment:
            raise RuntimeError("AZURE_OPENAI_DEPLOYMENT (or AZURE_OPENAI_MODEL) must be set for Azure provider")
        return AzureOpenAIWrapper(deployment=deployment)
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
