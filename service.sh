export PATH=$PATH:${PWD}/llama.cpp/build/bin/

export MODEL="${HOME}/GGUF/Qwen2.5-VL-3B-Instruct-GGUF/Qwen2.5-VL-3B-Instruct-Q8_0.gguf"
export MMPROJ="${HOME}/GGUF/Qwen2.5-VL-3B-Instruct-GGUF/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf"

#export MODEL="${PWD}/SmolVLM2-500M-Video-Instruct-GGUF/SmolVLM2-500M-Video-Instruct-Q8_0.gguf"
#export MMPROJ="${PWD}/SmolVLM2-500M-Video-Instruct-GGUF/mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf"

# VLM Service for AMD ROCm™
# This script starts the llama-server with the Qwen2.5-VL-3B-Instruct model
# optimized for AMD GPUs using ROCm™ hardware acceleration.
#
# Usage:
#   ./service.sh
#
# The service will start on port 8080 and can be accessed by demo.html
# or any client making requests to the /v1/chat/completions endpoint.
HSA_OVERRIDE_GFX_VERSION=11.0.0 llama-server \
	-m ${MODEL} \
	--mmproj ${MMPROJ} \
	--host 0.0.0.0 \
	--port 8080 \
	-fa \
	-ngl 99

#chromium-browser index.html
