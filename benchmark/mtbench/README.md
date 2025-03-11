## Download Dataset

```sh
wget -O question.jsonl https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl
```

## Run benchmark

### Benchmark sglang
```
autoregressive test:

Baseline:
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --port 30000 \
    --cuda-graph-max-bs 1 --mem-fraction 0.7 --dtype bfloat16 --enable-torch-compile



Baseline (no LoRA):
python3 -m sglang.launch_server --model /data/jamesliu/sglang/tulu/llama-3.1-instruct-phoenix_1layer_baseline/base_model  --speculative-algo PHOENIX \
    --speculative-draft /data/jamesliu/sglang/tulu/llama-3.1-instruct-phoenix_1layer_baseline/phoenix_model --speculative-num-steps 5 \
    --speculative-eagle-topk 8 --speculative-num-draft-tokens 64 \
    --cuda-graph-max-bs 1 --mem-fraction 0.7 --dtype bfloat16 --enable-torch-compile

LoRA:

python3 -m sglang.launch_server --model /data/jamesliu/sglang/tulu/llama-3.1-instruct-phoenix_1layer_lora/base_model  --speculative-algo PHOENIX \
    --speculative-draft /data/jamesliu/sglang/tulu/llama-3.1-instruct-phoenix_1layer_lora/phoenix_model --speculative-num-steps 6 \
    --speculative-eagle-topk 8 --speculative-num-draft-tokens 64 \
    --cuda-graph-max-bs 1 --mem-fraction 0.7 --dtype bfloat16 --speculative-phoenix-is-lora --enable-torch-compile


python3 -m sglang.launch_server --model /data/jamesliu/sglang/tulu/llama-3.1-instruct-phoenix_1layer_lora/base_model  --speculative-algo PHOENIX \
    --speculative-draft /data/jamesliu/sglang/tulu/llama-3.1-instruct-phoenix_1layer_lora/phoenix_model --speculative-num-steps 7 \
    --speculative-eagle-topk 16 --speculative-num-draft-tokens 128 \
    --cuda-graph-max-bs 1 --mem-fraction 0.7 --dtype bfloat16 --speculative-phoenix-is-lora --enable-torch-compile

k=1:

python3 -m sglang.launch_server --model meta-llama/Meta-Llama-3.1-8B-Instruct  --speculative-algo PHOENIX \
    --speculative-draft /data/jamesliu/sglang/phoenix_1layer_baseline/phoenix_model --speculative-num-steps 4 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --cuda-graph-max-bs 1 --mem-fraction 0.7 --dtype bfloat16

python3 -m sglang.launch_server --model /data/jamesliu/sglang/phoenix_1layer_lora/base_model  --speculative-algo PHOENIX \
    --speculative-draft /data/jamesliu/sglang/phoenix_1layer_lora/phoenix_model --speculative-num-steps 4 \
    --speculative-eagle-topk 8 --speculative-num-draft-tokens 8 \
    --cuda-graph-max-bs 1 --mem-fraction 0.7 --dtype bfloat16 \
    --speculative-phoenix-is-lora --enable-torch-compile

```

```
python3 bench_sglang.py --num-questions 80
```


### Benchmark vllm
```
python3 -m vllm.entrypoints.api_server --tokenizer-mode auto --model meta-llama/Llama-2-7b-chat-hf --disable-log-requests --port 21000
```

```
python3 bench_other.py --num-questions 80 --backend vllm
```


### Benchmark lightllm
```
# A10G
python -m lightllm.server.api_server --tokenizer_mode auto --model_dir ~/model_weights/llama-2-7b-chat-hf --max_total_token_num 16000 --port 22000
```

```
python3 bench_other.py --num-questions 80 --backend lightllm
```
