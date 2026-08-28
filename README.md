## Installation & Setup

### Option A: Batch Script (Faster)

**Requires:** Chocolatey for package management and Administrator privileges.
**Installs:** Python 3.12 & pip, Git & HuggingFace.Git-Xet, project dependencies, and the default LLM model listed below.
**Note:** This script assumes a fresh installation, but it can tolerate partial states up to a limit. It is not strictly unsafe, but it is a linear, fail-forward, one-shot process. It runs until it either finishes or you press `Ctrl+C`, which may leave the environment in an invalid state depending on when and where you interrupt it.

In your terminal of choice.
```
installer.bat
```
*It will echo the commands being used so you can monitor its progress.*

### Option B: Manual (Safer)

Follow these steps to set up the project environment. 

#### 1. Prerequisites

* **Python 3.12:** Download and install [Python 3.12 🐍](https://www.python.org/downloads/release/python-31210/). Ensure you check the box to **"Add Python to PATH"** during installation.
* **LLM File:** Download the [Meta-Llama-3-8B-Instruct-GGUF 🦙](https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF) model. Save it to `.\ai\Meta-Llama-3-8B-Instruct-Q3_K_M.gguf`, or replace it with a higher-end quantization if your GPU has more than 3 digits in its name, unlike mine.

#### 2. Clone the Repository

Choose **one** of the methods below to duplicate the codebase onto your local system or virtualized instance. 

##### Option A: Using the terminal

```
git clone https://github.com/Hjuldahr/SableResurrection.git && cd SableResurrection
```

##### Option B: Using Git CLI

```
gh repo clone Hjuldahr/SableResurrection`
```

##### Option C: Using VS Code UI

1. Ensure the [GitHub Repositories Extension](https://marketplace.visualstudio.com/items?itemName=GitHub.remotehub) is installed.
2. Press Ctrl + Shift + P (or Cmd + Shift + P on Mac) to open the Command Palette.
3. Type `Git: Clone` and press **Enter**.
4. Paste `https://github.com/Hjuldahr/SableResurrection.git` and select a local folder.

#### 3. Environment Setup

Choose **one** of the methods below to configure your virtual environment and install dependencies. 

##### Option A: Using the Command Line

1. Open your terminal in the project root folder.
2. Create the environment: 

If 3.12 is your newest installation:
```
python -m venv .venv
```
Otherwise:
```
py -3.12 -m venv .venv
```
3. Activate the environment: 
* **CMD:** 
```
.venv\Scripts\activate.bat
```
* **PowerShell:** 
```
.venv\Scripts\Activate.ps1
```
Using your standalone or IDE-integrated console

4. Install the required packages: 
```
pip install -r requirements.txt
```
##### Option B: Using VS Code UI

1. Ensure the [Python Extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python) is installed.
2. Press Ctrl + Shift + P (or ⌘ + Shift + P on Mac) to open the Command Palette.
3. Select **Python: Create Environment...**
4. Choose **Venv**.
5. Select **Python 3.12** as your interpreter.
6. When prompted to select a dependencies file, check **requirements.txt** and click **OK**.