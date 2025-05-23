python3 -m llama_cpp.server --model /path/to/your/model.gguf --n_gpu_layers 1

#1
brew install cmake pkg-config
#2
CMAKE_ARGS="-DGGML_METAL=on" pip install -U llama-cpp-python --no-cache-dir
pip install 'llama-cpp-python[server]'
