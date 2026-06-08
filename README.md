Local Ollama LLM (required for LLM features)
============================================

This project uses a locally hosted Ollama LLM accessed via the LangChain `Ollama` wrapper.

Key points
- The code expects an Ollama HTTP endpoint at `http://localhost:11434`.
- The model referenced in the code is `qwen2.5-coder:7b` (ensure the model is available locally).

Quick setup (high level)
1. Install Ollama for your OS: https://ollama.com/
2. Pull the model you want to use (example):

   ```powershell
   ollama pull qwen2.5-coder:7b
   ```

3. Start the Ollama service so it listens on the local API port (example):

   ```powershell
   ollama serve
   ```

   Ensure the service is reachable at `http://localhost:11434` (the code uses this base URL).

Verification
- Curl a health or models endpoint to confirm the service is running (example):

  ```powershell
  curl http://localhost:11434
  ```

Notes for this repo
- `app.py` and `diagnostics/orchestrator.py` initialize `Ollama(model=..., base_url="http://localhost:11434")`.
- If you prefer a hosted API (OpenAI/Azure/etc.), I can add a config toggle and fallback logic.

Troubleshooting
- If Streamlit shows "Could not reach local Ollama server", ensure Ollama is running and that any firewall allows local connections.

Configuration: external provider fallback
- By default the code uses Ollama at `http://localhost:11434`.
- To switch to an external OpenAI provider, set environment variables before running the app:

   ```powershell
   setx LLM_PROVIDER openai
   setx OPENAI_API_KEY <your_api_key>
   setx OPENAI_MODEL gpt-4o
   ```

- To keep using Ollama but change the model or URL:

   ```powershell
   setx LLM_PROVIDER ollama
   setx OLLAMA_MODEL qwen2.5-coder:7b
   setx OLLAMA_URL http://localhost:11434
   ```

- The code now uses a centralized provider helper `llm_provider.py` that reads `LLM_PROVIDER` and initializes the appropriate client.
