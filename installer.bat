choco upgrade python312 git -y
call refreshenv

winget install --id HuggingFace.Git-Xet -e --source winget --accept-source-agreements --accept-package-agreements
call refreshenv

git clone https://github.com/Hjuldahr/SableResurrection.git
cd SableResurrection

py -3.12 -m venv .venv
call .venv\Scripts\activate.bat

python -m pip install -r requirements.txt
python -m pip install "huggingface_hub[cli]"

mkdir ai 2>nul
hf download QuantFactory/Meta-Llama-3-8B-Instruct-GGUF Meta-Llama-3-8B-Instruct.Q3_K_M.gguf --local-dir ".\ai"
ren "ai\Meta-Llama-3-8B-Instruct.Q3_K_M.gguf" "Meta-Llama-3-8B-Instruct-Q3_K_M.gguf"