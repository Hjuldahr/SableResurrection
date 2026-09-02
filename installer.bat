:: installer
@echo on
setlocal

choco upgrade python312 git -y || exit /b 1 
call refreshenv

py -3.12 -m venv .venv || exit /b 1 
call .venv\Scripts\activate.bat

call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -vcvars_ver=14.29 || exit /b 1

python -m pip install --upgrade pip || exit /b 1

set "FORCE_CMAKE=1"
set "CUDACXX=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\nvcc.exe"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
set "CUDA_TOOLKIT_ROOT_DIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
set "CMAKE_ARGS=-G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=52"

python -m pip install -r requirements.txt || exit /b 1

python -m pip install llama-cpp-python --no-cache-dir --force-reinstall > build.log 2>&1 || exit /b 1

python -c "from llama_cpp import llama_supports_gpu_offload; raise SystemExit(0 if llama_supports_gpu_offload() else 1)" || (
    echo ERROR: llama-cpp-python was installed without CUDA support.
    exit /b 1
)

python -m pip install huggingface_hub || exit /b 1

mkdir llm 2>nul
:: requires auth to be allowed by hf services
set HF_XET_HIGH_PERFORMANCE=1 

hf auth login
hf download hf://bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --local-dir ".\llm" || exit 1
:: hf download hf://bartowski/microsoft_Phi-4-mini-instruct-GGUF/microsoft_Phi-4-mini-instruct-Q4_K_M.gguf --local-dir ".\llm" || exit 1
hf download hf://ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf --local-dir ".\llm"
hf download hf://ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf --local-dir ".\llm"

:: Remove Hugging Face download metadata/cache from the deployment
rmdir /s /q ".\llm\.cache" 2>nul


endlocal