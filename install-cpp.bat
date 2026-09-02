:: 1. Initialize the developer command prompt targeting the legacy 14.29 toolset
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -vcvars_ver=14.29

:: 3. Configure explicit build constraints for CMake and Ninja
set FORCE_CMAKE=1
set "CUDACXX=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\nvcc.exe"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
set "CUDA_TOOLKIT_ROOT_DIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
set "CMAKE_ARGS=-G Ninja -DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=52"

:: 4. Clear any broken wheels and run the clean source compilation
python -m pip uninstall llama-cpp-python -y
python -m pip install llama-cpp-python --no-cache-dir --force-reinstall >build.log 2>&1

:: 5. Sanity check the compilation instantly
python -c "import llama_cpp; print('Version:', llama_cpp.__version__)"
python -c "from llama_cpp import llama_supports_gpu_offload; print('GPU offload:', llama_supports_gpu_offload())"