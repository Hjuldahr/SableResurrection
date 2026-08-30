@echo off
set FORCE_CMAKE=1

REM === 1. INITIALIZE THE MSVC COMPILER ENVIRONMENT ===
if exist "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" (
    call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" amd64
) else if exist "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" (
    call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64
)

REM === 2. CONFIGURE COMPILER SETTINGS ===
:: -DCMAKE_CUDA_FLAGS overrides NVIDIA's compiler blocker
:: -DCMAKE_CXX_FLAGS overrides Microsoft's C++ Standard Library blocker
set "CMAKE_ARGS=-G "Ninja" -DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=52 -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler" -DCMAKE_CXX_FLAGS="-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH" -DCUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8" -DCUDAToolkit_ROOT="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8" -DCUDAToolkit_INCLUDE_DIR="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include" -DCUDAToolkit_LIBRARY_DIR="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib\x64""

set "CUDACXX=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\nvcc.exe"

REM === 3. EXECUTE BUILD ===
echo [INFO] Starting compilation. Bypassing BOTH Microsoft and NVIDIA version locks...
pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir 2> error.log

if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] Hardware acceleration build complete!
) else (
    echo [FAILED] Compilation failed. Let's see what the new blocker is.
)

pause