#!/usr/bin/env bash
set -euo pipefail

# The checkpoint and tokenizer must resolve from attached Kaggle Inputs only.
# These defaults make a direct launcher invocation offline as well as the
# notebook path; an explicit caller override is still respected.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
E1_MODEL_ID="${E1_MODEL_ID:-/kaggle/input/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1}"
E1_SERVED_MODEL_NAME="${E1_SERVED_MODEL_NAME:-$E1_MODEL_ID}"
E1_METADATA_PATH="${E1_METADATA_PATH:-e1_model_metadata.json}"
if [[ ! -d "$E1_MODEL_ID" ]]; then
  echo "E1_MODEL_ID must be an existing HF checkpoint directory: $E1_MODEL_ID" >&2
  exit 2
fi
E1_SERVER_HOST="${E1_SERVER_HOST:-127.0.0.1}"
E1_SERVER_PORT="${E1_SERVER_PORT:-1234}"
E1_TP="${E1_TENSOR_PARALLEL_SIZE:-1}"
E1_MAX_MODEL_LEN="${E1_MAX_MODEL_LEN:-32768}"
E1_GPU_MEMORY_UTILIZATION="${E1_GPU_MEMORY_UTILIZATION:-0.92}"
E1_TEMPERATURE="${E1_TEMPERATURE:-0.6}"
E1_TOP_P="${E1_TOP_P:-0.95}"
E1_TOP_K="${E1_TOP_K:-20}"
E1_MAX_TOKENS="${E1_MAX_TOKENS:-2048}"
E1_ENABLE_THINKING="${E1_ENABLE_THINKING:-true}"
E1_PRESERVE_THINKING="${E1_PRESERVE_THINKING:-true}"
E1_REASONING_PARSER="${E1_REASONING_PARSER:-qwen3}"
E1_TOOL_CALL_PARSER="${E1_TOOL_CALL_PARSER:-qwen3_coder}"

# Print and persist local checkpoint metadata before vLLM loads weights. The
# file records UNKNOWN for fields absent from config/tokenizer files.
E1_MODEL_ID="$E1_MODEL_ID" \
E1_SERVED_MODEL_NAME="$E1_SERVED_MODEL_NAME" \
E1_METADATA_PATH="$E1_METADATA_PATH" \
E1_BASE_URL="http://$E1_SERVER_HOST:$E1_SERVER_PORT/v1" \
E1_MAX_MODEL_LEN="$E1_MAX_MODEL_LEN" \
E1_TEMPERATURE="$E1_TEMPERATURE" E1_TOP_P="$E1_TOP_P" E1_TOP_K="$E1_TOP_K" \
E1_MAX_TOKENS="$E1_MAX_TOKENS" E1_ENABLE_THINKING="$E1_ENABLE_THINKING" \
E1_PRESERVE_THINKING="$E1_PRESERVE_THINKING" E1_REASONING_PARSER="$E1_REASONING_PARSER" \
E1_TOOL_CALL_PARSER="$E1_TOOL_CALL_PARSER" \
python "$SCRIPT_DIR/model_metadata.py" --model-path "$E1_MODEL_ID" \
  --served-model-name "$E1_SERVED_MODEL_NAME" --output "$E1_METADATA_PATH"

# vLLM renamed the server-side chat-template kwargs flag across releases.
# The request path in OpenAICompatibleBackend always sends chat_template_kwargs;
# this optional server flag only supplies a default for clients that omit it.
E1_CHAT_TEMPLATE_KWARGS="${E1_CHAT_TEMPLATE_KWARGS:-{\"enable_thinking\":true,\"preserve_thinking\":true}}"
VLLM_MODULE="vllm.entrypoints.openai.api_server"
VLLM_KWARGS=()
VLLM_HELP="$(python -m "$VLLM_MODULE" --help 2>&1 || true)"
if [[ "$VLLM_HELP" == *"--default-chat-template-kwargs"* ]]; then
  VLLM_KWARGS+=(--default-chat-template-kwargs "$E1_CHAT_TEMPLATE_KWARGS")
elif [[ "$VLLM_HELP" == *"--chat-template-kwargs"* ]]; then
  VLLM_KWARGS+=(--chat-template-kwargs "$E1_CHAT_TEMPLATE_KWARGS")
fi
if [[ "$VLLM_HELP" == *"--tool-call-parser"* ]]; then
  VLLM_KWARGS+=(--tool-call-parser "$E1_TOOL_CALL_PARSER")
fi

exec python -m "$VLLM_MODULE" \
  --model "$E1_MODEL_ID" \
  --served-model-name "$E1_SERVED_MODEL_NAME" \
  --host "$E1_SERVER_HOST" \
  --port "$E1_SERVER_PORT" \
  --tensor-parallel-size "$E1_TP" \
  --dtype auto \
  --max-model-len "$E1_MAX_MODEL_LEN" \
  --gpu-memory-utilization "$E1_GPU_MEMORY_UTILIZATION" \
  --trust-remote-code \
  --enable-prefix-caching \
  --reasoning-parser "$E1_REASONING_PARSER" \
  "${VLLM_KWARGS[@]}"
