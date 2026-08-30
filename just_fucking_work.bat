@echo off
set FORCE_CMAKE=1

if exist "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" (
    call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" amd64
) else if exist "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" (
    call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64
)

:: Adjusted paths for v12.1 while keeping the compute_52 force injection alive
set "CMAKE_ARGS=-G "Ninja" -DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=52 -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler" -DCMAKE_CXX_FLAGS="-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH" -DCUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1" -DCUDAToolkit_ROOT="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1" -DCUDAToolkit_INCLUDE_DIR="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\include" -DCUDAToolkit_LIBRARY_DIR="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\lib\x64""

set "CUDACXX=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin\nvcc.exe"

echo [INFO] Compiling with CUDA 12.1 + Architecture Override...
pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir 2> error.log
pause