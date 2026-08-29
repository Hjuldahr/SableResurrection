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

set HF_XET_HIGH_PERFORMANCE=1 
hf download QuantFactory/Meta-Llama-3-8B-Instruct-GGUF --include "Meta-Llama-3-8B-Instruct.Q3_K_M.gguf" --local-dir ".\llm" || exit 1
:: Remove Hugging Face download metadata/cache from the deployment
rmdir /s /q ".\llm\.cache" 2>nul

endlocal