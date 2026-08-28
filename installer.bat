@echo on
setlocal

choco upgrade python312 git -y || exit 1 
call refreshenv

git clone https://github.com/Hjuldahr/SableResurrection.git || exit 1
cd SableResurrection

py -3.12 -m venv .venv
call .venv\Scripts\activate.bat

python -m pip install --upgrade pip
python -m pip install -r requirements.txt || exit 1
python -m pip install "huggingface_hub[cli]" || exit 1

mkdir ai 2>nul
hf download QuantFactory/Meta-Llama-3-8B-Instruct-GGUF Meta-Llama-3-8B-Instruct.Q3_K_M.gguf --local-dir ".\ai" || exit 1
ren "ai\Meta-Llama-3-8B-Instruct.Q3_K_M.gguf" "Meta-Llama-3-8B-Instruct-Q3_K_M.gguf"

endlocal