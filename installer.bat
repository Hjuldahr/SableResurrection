:: installer
@echo on
setlocal

choco upgrade python312 git -y || exit /b 1 
call refreshenv

py -3.12 -m venv .venv || exit /b 1 
call .venv\Scripts\activate.bat

call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -vcvars_ver=14.29 || exit /b 1

python -m pip install --upgrade pip || exit /b 1
python -m pip install -r requirements.txt || exit /b 1
python -m pip install huggingface_hub || exit /b 1

mkdir llm 2>nul
:: requires auth to be allowed by hf services
set HF_XET_HIGH_PERFORMANCE=1 

hf auth login
hf download hf://bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --local-dir ".\llm" || exit 1
hf download hf://ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf --local-dir ".\llm"
hf download hf://ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf --local-dir ".\llm"

ollama create qwen -f ./QwenModelFile

:: Remove Hugging Face download metadata/cache from the deployment
rmdir /s /q ".\llm\.cache" 2>nul

endlocal