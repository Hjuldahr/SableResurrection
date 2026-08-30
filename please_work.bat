set FORCE_CMAKE=1
set CUDACXX=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\nvcc.exe
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
set CUDA_TOOLKIT_ROOT_DIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
set CMAKE_ARGS=-G Ninja -DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=52

python -m pip uninstall llama-cpp-python -y

python -m pip install llama-cpp-python --no-cache-dir --force-reinstall > build.log 2>&1