MODEL_ARGS=(
   --spec "slime_plugins.models.olmo3" "get_olmo3_spec"

   --swiglu
   --num-layers 32
   --hidden-size 4096
   --ffn-hidden-size 11008
   --num-attention-heads 32
   --group-query-attention
   --num-query-groups 32
   --kv-channels 128

   --disable-bias-linear
   --normalization RMSNorm
   --norm-epsilon 1e-6
   --untie-embeddings-and-output-weights
   --vocab-size 100278
   --make-vocab-size-divisible-by 1

   --position-embedding-type none
   --rotary-percent 1.0
   --rotary-base 500000
   --max-position-embeddings 65536

   # OLMo 3 alternates three sliding layers with one full-attention layer.
   --window-size 4096,0
   --window-attn-skip-freq 4

   # Bound temporary [tokens, vocab] buffers for the 100,278-token vocabulary.
   # Chunking is differentiable and leaves log-prob/entropy numerics unchanged.
   --log-probs-chunk-size 2048
)
