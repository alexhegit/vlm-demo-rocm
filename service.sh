export PATH=$PATH:${HOME}/llamacpp-vulkan-b7132/build/bin/

#export HSA_OVERRIDE_GFX_VERSION=11.0.0

#export MODEL="${HOME}/GGUF/Qwen3VL-8B-Thinking-GGUF/Qwen3VL-8B-Thinking-Q4_K_M.gguf"
#export MMPROJ="${HOME}/GGUF/Qwen3VL-8B-Thinking-GGUF/mmproj-Qwen3VL-8B-Thinking-Q8_0.gguf"

export MODEL="${HOME}/GGUF/SmolVLM2-500M-Video-Instruct-GGUF/SmolVLM2-500M-Video-Instruct-Q8_0.gguf"
export MMPROJ="${HOME}/GGUF/SmolVLM2-500M-Video-Instruct-GGUF/mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf"

# VLM Service for AMD ROCm™
# This script starts the llama-server with the Qwen2.5-VL-3B-Instruct model
# optimized for AMD GPUs using ROCm™ hardware acceleration.
#
# Usage:
#   ./service.sh
#
# The service will start on port 8080 and can be accessed by demo.html
# or any client making requests to the /v1/chat/completions endpoint.

llama-server \
	-m ${MODEL} \
	--mmproj ${MMPROJ} \
	--host 0.0.0.0 \
	--port 8080 \
	-fa on \
	-ngl 99

#chromium-browser index.html
