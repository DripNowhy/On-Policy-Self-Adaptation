MODEL_DISPLAY_NAME="OLMo-3-7B-Instruct"
MODEL_CONFIG_RELATIVE="scripts/models/olmo3-7B-Instruct.sh"
NUM_ROLLOUT=1000
ACTOR_GPUS=4
ROLLOUT_GPUS=4
# OLMo 3 normalizes complete Q/K projections, so TP remains one. Split its
# layers over two pipeline stages to leave room for differentiable 100k logits.
TENSOR_MODEL_PARALLEL_SIZE=1
PIPELINE_MODEL_PARALLEL_SIZE=2
ROLLOUT_GPUS_PER_ENGINE=1
ROLLOUT_MAX_RESPONSE_LEN=16384
EVAL_MAX_RESPONSE_LEN=32768
# OLMo's 100k vocabulary makes differentiable logits materially larger than
# Qwen's; cap dynamic micro-batches to keep the log-prob/entropy peak in memory.
MAX_TOKENS_PER_GPU=16384
SGLANG_MEM_FRACTION_STATIC=0.82
SGLANG_CUDA_GRAPH_MAX_BS=64
OPTIMIZER_CPU_OFFLOAD=true
DISABLE_THINKING=false
