:: Deinstaller
@echo on

set "TARGET_DIR=%~dp0"

if exist "%TARGET_DIR%.venv\Scripts\hf.exe" (
    call "%TARGET_DIR%.venv\Scripts\hf.exe" cache rm model/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF -y
)

if exist "%TARGET_DIR%.venv\Scripts\deactivate.bat" (
    call "%TARGET_DIR%.venv\Scripts\deactivate.bat"
)

set "DELETE_PATH=%TARGET_DIR%"

cd /d %TEMP%

start "" /b cmd /c "timeout /t 1 >nul & rmdir /s /q \"%TARGET_DIR%\""
exit /b