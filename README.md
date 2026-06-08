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

- To use Azure OpenAI, set the following environment variables (example):
 - To use Azure OpenAI, set the following environment variables (example):

   ```powershell
   setx LLM_PROVIDER azure
   setx AZURE_OPENAI_ENDPOINT https://your-resource-name.openai.azure.com/
   setx AZURE_OPENAI_KEY <your_api_key>
   setx AZURE_OPENAI_DEPLOYMENT <deployment_name>
   ```

 - Native SDK (preferred): this project prefers the native `azure-ai-openai` SDK when available. Install it with:

   ```powershell
   pip install azure-ai-openai
   ```

   When installed, the app will use `azure.ai.openai.OpenAIClient` (recommended). If the native SDK is not installed, the code falls back to using the `openai` package configured for Azure.

VS Code / Pylance troubleshooting
- If you see `Import "azure.ai.openai" could not be resolved` (Pylance), ensure the package is installed into the Python interpreter selected by VS Code:

   1. In VS Code, open the Command Palette and run `Python: Select Interpreter` — choose the environment you use to run the app.
   2. Install the native SDK (or `openai`) into that environment, for example:

       ```powershell
       pip install -r requirements.txt
       pip install azure-ai-openai
       ```

   3. Restart VS Code or run the `Developer: Reload Window` command.

- Alternative: keep imports dynamic (already implemented) — the app will only raise an error at runtime if the native SDK is missing when `LLM_PROVIDER=azure`.

- If Pylance still flags unresolved imports, you can configure `python.analysis.extraPaths` in `.vscode/settings.json` to point to your environment's site-packages folder, or ensure your workspace uses the same interpreter.

Developer setup
1. Create a virtual environment and activate it (PowerShell):

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install runtime and development dependencies:

   ```powershell
   pip install -r requirements.txt
   pip install -r dev-requirements.txt
   ```

3. Install and enable pre-commit hooks:

   ```powershell
   pre-commit install
   pre-commit run --all-files
   ```

CI
- A GitHub Actions workflow is included at `.github/workflows/ci.yml` that runs `black --check`, `isort --check-only`, `flake8`, `mypy`, and `pytest` on push and pull requests to `main`.
