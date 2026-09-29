MODEL_DISPLAY_NAME="OLMo-3-7B-Instruct-SFT"
MODEL_CONFIG_RELATIVE="scripts/models/olmo3-7B-Instruct.sh"
NUM_ROLLOUT=1000
ACTOR_GPUS=4
ROLLOUT_GPUS=4

# OLMo 3 applies Q/K RMSNorm over complete projections. Keep TP=1 for exact
# semantics and split the 32 transformer layers across two pipeline stages.
TENSOR_MODEL_PARALLEL_SIZE=1
PIPELINE_MODEL_PARALLEL_SIZE=2
ROLLOUT_GPUS_PER_ENGINE=1

ROLLOUT_MAX_RESPONSE_LEN=10000
EVAL_MAX_RESPONSE_LEN=32768

# Its 100,278-token vocabulary makes differentiable logits large. Chunking in
# the model definition plus this micro-batch cap keeps entropy/log-prob memory
# bounded without changing OPSA token selection.
MAX_TOKENS_PER_GPU=16384
SGLANG_MEM_FRACTION_STATIC=0.82
SGLANG_CUDA_GRAPH_MAX_BS=64
OPTIMIZER_CPU_OFFLOAD=true

# OLMo uses its native instruct template and has no Qwen enable_thinking flag.
DISABLE_THINKING=false
