## Download Dataset

```sh
wget -O question.jsonl https://raw.githubusercontent.com/hemingkx/Spec-Bench/main/data/spec_bench/question.jsonl

```

## Run benchmark

### Benchmark sglang
```


Testing autoregressive no-op:
zero python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --port 30000
zero python -m sglang.launch_server --model-path /data/jamesliu/sglang/llama-3.1-8b-instruct-phoenix_1layer_baseline/base_model --port 30000


Testing LoRA no-op:
zero python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --port 30000
zero python -m sglang.launch_server --model-path /data/jamesliu/sglang/llama-3.1-8b-instruct-phoenix_1layer_lora/base_model --port 30000

Specdec baseline:

python3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct  --speculative-algo PHOENIX \
    --speculative-draft /data/jamesliu/sglang/llama-3.1-8b-instruct-phoenix_1layer_baseline/phoenix_model --speculative-num-steps 5 \
    --speculative-eagle-topk 8 --speculative-num-draft-tokens 64 \
    --cuda-graph-max-bs 1 --mem-fraction 0.7 --dtype bfloat16

Specdec LoRA no-op:

python3 -m sglang.launch_server --model-path /data/jamesliu/sglang/llama-3.1-8b-instruct-phoenix_1layer_lora/base_model  \
    --speculative-algo PHOENIX     --speculative-draft /data/jamesliu/sglang/llama-3.1-8b-instruct-phoenix_1layer_baseline/phoenix_model \
    --speculative-num-steps 5     --speculative-eagle-topk 8 --speculative-num-draft-tokens 128     \
    --cuda-graph-max-bs 1 --mem-fraction 0.7 --dtype bfloat16     --speculative-phoenix-is-lora

Specdec LoRA:

python3 -m sglang.launch_server --model-path /data/jamesliu/sglang/llama-3.1-8b-instruct-phoenix_1layer_lora/base_model  \
    --speculative-algo PHOENIX     --speculative-draft /data/jamesliu/sglang/llama-3.1-8b-instruct-phoenix_1layer_lora/phoenix_model \
    --speculative-num-steps 5     --speculative-eagle-topk 8 --speculative-num-draft-tokens 128     \
    --cuda-graph-max-bs 1 --mem-fraction 0.7 --dtype bfloat16     --speculative-phoenix-is-lora


```