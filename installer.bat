@echo off
setlocal

net session >nul 2>&1
if %errorLevel% neq 0 (
    echo Error: You must run this script as an Administrator!
    pause
    exit /b 1
)

for %%I in ("%CD%") do set "CURR_DIR=%%~nxI"

:: Check if already in the project directory
if /I "%CURR_DIR%"=="Sable" goto :project_ready
if /I "%CURR_DIR%"=="SableResurrection" goto :project_ready

:: Install Chocolatey
if not defined ChocolateyInstall (
    echo Installing Chocolatey...
    @"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -InputFormat None -ExecutionPolicy Bypass -Command "[System.Net.ServicePointManager]::SecurityProtocol = 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))"

    if %errorlevel% neq 0 (
        echo Error: Failed to install Chocolatey.
        pause
        exit /b 1
    )

    set "PATH=%PATH%;%ALLUSERSPROFILE%\chocolatey\bin"
)

:: Ensure the environment is up to date
call refreshenv

where choco >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Chocolatey is not available.
    pause
    exit /b 1
)

:: Check for Python 3.12
py -3.12 --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python 3.12 not found. Installing via Chocolatey...
    choco install python312 -y

    if %errorlevel% neq 0 (
        echo Error: Failed to install Python 3.12.
        pause
        exit /b 1
    )
)

:: Check for Git
where git >nul 2>&1
if %errorlevel% neq 0 (
    echo Git not found. Installing via Chocolatey...
    choco install git -y
    :: choco did not work for it
    winget install git-xet -y

    if %errorlevel% neq 0 (
        echo Error: Failed to install Git.
        pause
        exit /b 1
    )
)

:: Ensure the environment is up to date
call refreshenv

:: Clone repository if necessary
if not exist "SableResurrection\.git" (
    if exist "SableResurrection" (
        echo Error: SableResurrection exists but is not a Git repository.
        pause
        exit /b 1
    )

    echo Cloning SableResurrection...
    git clone https://github.com/Hjuldahr/SableResurrection.git

    if %errorlevel% neq 0 (
        echo Error: Failed to clone repository.
        pause
        exit /b 1
    )
)

cd /d SableResurrection
if %errorlevel% neq 0 exit /b 1

:project_ready

echo Configuring Virtual Environment...

if not exist ".venv\Scripts\activate.bat" (
    py -3.12 -m venv .venv

    if %errorlevel% neq 0 (
        echo Error: Failed to create virtual environment.
        pause
        exit /b 1
    )
)

call .venv\Scripts\activate.bat

echo Upgrading pip and installing requirements...

python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo Error: Failed to upgrade pip.
    pause
    exit /b 1
)

if exist "requirements.txt" (
    python -m pip install -r requirements.txt

    if %errorlevel% neq 0 (
        echo Error: Failed to install requirements.
        pause
        exit /b 1
    )
) else (
    echo requirements.txt not found, manual installation required.
)

set "TARGET_DIR=%CD%\ai"
set "SOURCE_FILE=Meta-Llama-3-8B-Instruct.Q3_K_M.gguf"
set "TARGET_FILE=Meta-Llama-3-8B-Instruct-Q3_K_M.gguf"
set "SOURCE_PATH=%TARGET_DIR%\%SOURCE_FILE%"
set "TARGET_PATH=%TARGET_DIR%\%TARGET_FILE%"

if not exist "%TARGET_PATH%" (
    echo Downloading %TARGET_FILE% 🦙 from QuantFactory/Meta-Llama-3-8B-Instruct-GGUF @ Hugging Face 🤗

    :: Ensure target directory exists
    if not exist "%TARGET_DIR%" mkdir "%TARGET_DIR%"

    :: Install Hugging Face CLI if not already installed
    where hf >nul 2>&1
    if errorlevel 1 (
        echo Installing Hugging Face CLI...
        powershell -ExecutionPolicy Bypass -Command "Invoke-RestMethod https://hf.co/cli/install.ps1 | Invoke-Expression"

        if errorlevel 1 (
            echo Error: Failed to install Hugging Face CLI.
            pause
            exit /b 1
        )

        call refreshenv
    )

    :: Download only the specific file directly to the target folder
    huggingface-cli download QuantFactory/Meta-Llama-3-8B-Instruct-GGUF "%SOURCE_FILE%" --local-dir "%TARGET_DIR%"

    if errorlevel 1 (
        echo.
        echo File failed to download. Either the file no longer exists,
        echo network issues occurred, or Hugging Face is experiencing downtime.
        echo Please check the repository structure or download it manually.
    ) else (
        ren "%SOURCE_PATH%" "%TARGET_FILE%"

        if errorlevel 1 (
            echo.
            echo Error: Download succeeded, but the file could not be renamed.
            pause
            exit /b 1
        )

        echo Success! File saved to %TARGET_DIR%
    )
) else (
    echo File already exists at %TARGET_PATH%
)

echo.
echo Project ready!
echo Python:
python --version
echo Virtual environment:
where python

endlocal