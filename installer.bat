:: installer
@echo on
setlocal

choco upgrade python312 git -y || exit 1 
call refreshenv

py -3.12 -m venv .venv || exit 1 
call .venv\Scripts\activate.bat

python -m pip install --upgrade pip



python -m pip install -r requirements.txt || exit 1

python -m pip install huggingface_hub || exit 1

mkdir llm 2>nul
:: requires auth to be allowed by hf services
set HF_XET_HIGH_PERFORMANCE=1 

hf auth login
hf download hf://bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --local-dir ".\llm" || exit 1
hf download hf://bartowski/microsoft_Phi-4-mini-instruct-GGUF/microsoft_Phi-4-mini-instruct-Q4_K_M.gguf --local-dir ".\llm" || exit 1
:: Remove Hugging Face download metadata/cache from the deployment
rmdir /s /q ".\llm\.cache" 2>nul


endlocal