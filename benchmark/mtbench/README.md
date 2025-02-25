## Download Dataset

```sh
wget -O question.jsonl https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl
```

## Run benchmark

### Benchmark sglang
```


Final:
python3 -m sglang.launch_server --model /data/jamesliu/sglang/phoenix_1layer_lora/base_model  --speculative-algo PHOENIX \
    --speculative-draft /data/jamesliu/sglang/phoenix_1layer_lora/phoenix_model --speculative-num-steps 5 \
    --speculative-eagle-topk 8 --speculative-num-draft-tokens 128 \
    --cuda-graph-max-bs 1 --mem-fraction 0.7 --dtype bfloat16 \
    --speculative-phoenix-is-lora

python3 -m sglang.launch_server --model /data/jamesliu/sglang/phoenix_1layer_lora/base_model  --speculative-algo PHOENIX \
    --speculative-draft /data/jamesliu/sglang/phoenix_1layer_lora/phoenix_model --speculative-num-steps 5 \
    --speculative-eagle-topk 8 --speculative-num-draft-tokens 128 \
    --disable-cuda-graph --mem-fraction 0.7 --dtype bfloat16 \
    --speculative-phoenix-is-lora



python3 -m sglang.launch_server --model /data/jamesliu/sglang/phoenix_1layer_lora/base_model  --speculative-algo PHOENIX \
    --speculative-draft /data/jamesliu/sglang/phoenix_1layer_baseline/phoenix_model --speculative-num-steps 5 \
    --speculative-eagle-topk 8 --speculative-num-draft-tokens 128 \
    --cuda-graph-max-bs 1 --mem-fraction 0.7 --dtype bfloat16 \
    --speculative-phoenix-is-lora


Linear no op:
python3 -m sglang.launch_server --model /data/jamesliu/sglang/phoenix_1layer_lora/base_model  --speculative-algo PHOENIX \
    --speculative-draft /data/jamesliu/sglang/phoenix_1layer_baseline/phoenix_model --speculative-num-steps 5 \
    --speculative-eagle-topk 8 --speculative-num-draft-tokens 128 \
    --cuda-graph-max-bs 1 --mem-fraction 0.7 --dtype bfloat16 \
    --speculative-phoenix-is-lora





python3 -m sglang.launch_server --model meta-llama/Meta-Llama-3.1-8B-Instruct  --speculative-algo PHOENIX \
    --speculative-draft /data/jamesliu/sglang/phoenix_1layer_baseline/phoenix_model --speculative-num-steps 5 \
    --speculative-eagle-topk 8 --speculative-num-draft-tokens 64 \
    --cuda-graph-max-bs 1 --mem-fraction 0.7 --dtype bfloat16



python3 -m sglang.launch_server --model /data/jamesliu/sglang/phoenix_1layer_lora/base_model  --speculative-algo PHOENIX \
    --speculative-draft /data/jamesliu/sglang/phoenix_1layer_baseline/phoenix_model --speculative-num-steps 5 \
    --speculative-eagle-topk 8 --speculative-num-draft-tokens 128 \
    --cuda-graph-max-bs 1 --mem-fraction 0.7 --dtype bfloat16 \
    --speculative-phoenix-is-lora


python3 -m sglang.launch_server --model /data/jamesliu/sglang/phoenix_1layer_lora/base_model  --speculative-algo PHOENIX \
    --speculative-draft /data/jamesliu/sglang/phoenix_1layer_baseline/phoenix_model --speculative-num-steps 5 \
    --speculative-eagle-topk 8 --speculative-num-draft-tokens 128 \
    --cuda-graph-max-bs 1 --mem-fraction 0.7 --dtype bfloat16 \
    --speculative-phoenix-is-lora


Test 1:
python3 -m sglang.launch_server --model /data/jamesliu/sglang/phoenix_1layer_lora/base_model  --speculative-algo PHOENIX \
    --speculative-draft /data/jamesliu/sglang/phoenix_1layer_lora/phoenix_model --speculative-num-steps 5 \
    --speculative-eagle-topk 8 --speculative-num-draft-tokens 128 \
    --cuda-graph-max-bs 1 --mem-fraction 0.7 --dtype bfloat16 \
    --speculative-phoenix-is-lora

Test 2:
python3 -m sglang.launch_server --model meta-llama/Meta-Llama-3.1-8B-Instruct  --speculative-algo PHOENIX \
    --speculative-draft /data/jamesliu/sglang/phoenix_1layer_baseline/phoenix_model --speculative-num-steps 5 \
    --speculative-eagle-topk 8 --speculative-num-draft-tokens 64 \
    --cuda-graph-max-bs 1 --mem-fraction 0.7 --dtype bfloat16



Phoenix:

python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --port 30000 --cuda-graph-max-bs 1 --dtype bfloat16 

python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --port 30000 --cuda-graph-max-bs 1 --dtype bfloat16 --enable-torch-compile


LoRA:

python -m sglang.launch_server --model-path /data/jamesliu/sglang/phoenix_1layer_lora/base_model --port 30000 --disable-cuda-graph --dtype bfloat16 

python -m sglang.launch_server --model-path /data/jamesliu/sglang/phoenix_1layer_lora/base_model --port 30000 --cuda-graph-max-bs 1 --dtype bfloat16 

python -m sglang.launch_server --model-path /data/jamesliu/sglang/phoenix_1layer_lora/base_model --port 30000 --cuda-graph-max-bs 1 --dtype bfloat16 --enable-torch-compile



python3 -m sglang.launch_server --model /data/jamesliu/sglang/phoenix_1layer_lora/base_model  --speculative-algo PHOENIX \
    --speculative-draft /data/jamesliu/sglang/phoenix_1layer_baseline/phoenix_model --speculative-num-steps 5 \
    --speculative-eagle-topk 8 --speculative-num-draft-tokens 64 \
    --disable-cuda-graph --mem-fraction 0.7 --dtype bfloat16 \
    --speculative-phoenix-is-lora






No-op (on the lora linear):

python3 -m sglang.launch_server --model /data/jamesliu/sglang/phoenix_1layer_lora/base_model  --speculative-algo PHOENIX \
    --speculative-draft /data/jamesliu/sglang/phoenix_1layer_lora/phoenix_model --speculative-num-steps 5 \
    --speculative-eagle-topk 8 --speculative-num-draft-tokens 64 \
    --cuda-graph-max-bs 1 --mem-fraction 0.7 --dtype bfloat16 \
    --speculative-phoenix-is-lora



No-op 1 (on the verify process):
python3 -m sglang.launch_server --model /data/jamesliu/sglang/phoenix_1layer_lora/base_model  --speculative-algo PHOENIX \
    --speculative-draft /data/jamesliu/sglang/phoenix_1layer_baseline/phoenix_model --speculative-num-steps 5 \
    --speculative-eagle-topk 8 --speculative-num-draft-tokens 64 \
    --disable-cuda-graph --mem-fraction 0.7 --dtype bfloat16 \
    --speculative-phoenix-is-lora

No-op 2:

python3 -m sglang.launch_server --model meta-llama/Meta-Llama-3.1-8B-Instruct  --speculative-algo PHOENIX \
    --speculative-draft /data/jamesliu/sglang/phoenix_1layer_baseline/phoenix_model --speculative-num-steps 5 \
    --speculative-eagle-topk 8 --speculative-num-draft-tokens 32 \
    --disable-cuda-graph --mem-fraction 0.7 --dtype bfloat16

No-op 3:

python3 -m sglang.launch_server --model meta-llama/Meta-Llama-3.1-8B-Instruct  --speculative-algo EAGLE \
    --speculative-draft /data/jamesliu/sglang/phoenix_1layer_baseline/phoenix_model --speculative-num-steps 5 \
    --speculative-eagle-topk 8 --speculative-num-draft-tokens 32 \
    --disable-cuda-graph --mem-fraction 0.7 --dtype bfloat16



python3 -m sglang.launch_server --model /data/jamesliu/sglang/phoenix_1layer_lora/base_model  --speculative-algo PHOENIX \
    --speculative-draft /data/jamesliu/sglang/phoenix_1layer_baseline/phoenix_model --speculative-num-steps 5 \
    --speculative-eagle-topk 8 --speculative-num-draft-tokens 64 \
    --cuda-graph-max-bs 1 --mem-fraction 0.7 --dtype bfloat16 \
    --speculative-phoenix-is-lora --enable-torch-compile


python3 -m sglang.launch_server --model meta-llama/Meta-Llama-3.1-8B-Instruct  --speculative-algo PHOENIX \
    --speculative-draft /data/jamesliu/sglang/phoenix_1layer_lora/phoenix_model --speculative-num-steps 5 \
    --speculative-eagle-topk 8 --speculative-num-draft-tokens 32 \
    --disable-cuda-graph --mem-fraction 0.7 --dtype bfloat16


Non-LoRA:
python3 -m sglang.launch_server --model meta-llama/Meta-Llama-3.1-8B-Instruct  --speculative-algo PHOENIX \
    --speculative-draft /data/jamesliu/sglang/phoenix_1layer_baseline/phoenix_model --speculative-num-steps 5 \
    --speculative-eagle-topk 7 --speculative-num-draft-tokens 8 \
    --disable-cuda-graph --mem-fraction 0.7 --dtype bfloat16


python3 -m sglang.launch_server --model meta-llama/Meta-Llama-3.1-8B-Instruct  --speculative-algo PHOENIX \
    --speculative-draft /data/jamesliu/sglang/phoenix_1layer_baseline/phoenix_model --speculative-num-steps 5 \
    --speculative-eagle-topk 8 --speculative-num-draft-tokens 64 \
    --cuda-graph-max-bs 1 --mem-fraction 0.7 --dtype bfloat16



SHITTY:
python3 -m sglang.launch_server --model meta-llama/Meta-Llama-3.1-8B-Instruct  --speculative-algo PHOENIX \
    --speculative-draft /data/jamesliu/sglang/phoenix_1layer_baseline/phoenix_model --speculative-num-steps 5 \
    --speculative-eagle-topk 8 --speculative-num-draft-tokens 64 \
    --cuda-graph-max-bs 1 --mem-fraction 0.7 --dtype bfloat16 --disable-cuda-graph


python3 -m sglang.launch_server --model meta-llama/Meta-Llama-3-8B-Instruct  --speculative-algo EAGLE \
    --speculative-draft lmzheng/sglang-EAGLE-LLaMA3-Instruct-8B --speculative-num-steps 5 \
    --speculative-eagle-topk 8 --speculative-num-draft-tokens 64 \
    --cuda-graph-max-bs 1 --mem-fraction 0.7 --dtype float16 --disable-cuda-graph

python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --port 30000

python3 -m sglang.launch_server --model meta-llama/Meta-Llama-3-8B-Instruct  --speculative-algo EAGLE \
    --speculative-draft lmzheng/sglang-EAGLE-LLaMA3-Instruct-8B --speculative-num-steps 5 \
    --speculative-eagle-topk 8 --speculative-num-draft-tokens 64 \
    --cuda-graph-max-bs 1 --mem-fraction 0.7 --dtype float16

python3 -m sglang.launch_server --model meta-llama/Meta-Llama-3-8B-Instruct  --speculative-algo EAGLE \
    --speculative-draft lmzheng/sglang-EAGLE-LLaMA3-Instruct-8B --speculative-num-steps 5 \
    --speculative-eagle-topk 16 --speculative-num-draft-tokens 64 \
    --cuda-graph-max-bs 1 --mem-fraction 0.7 --dtype bfloat16


python3 -m sglang.launch_server --model meta-llama/Meta-Llama-3-8B-Instruct  --speculative-algo EAGLE \
    --speculative-draft lmzheng/sglang-EAGLE-LLaMA3-Instruct-8B --speculative-num-steps 5 \
    --speculative-eagle-topk 4 --speculative-num-draft-tokens 32 \
    --cuda-graph-max-bs 1 --mem-fraction 0.7 --dtype bfloat16


python3 -m sglang.launch_server --model meta-llama/Meta-Llama-3-8B-Instruct  --speculative-algo EAGLE \
    --speculative-draft lmzheng/sglang-EAGLE-LLaMA3-Instruct-8B --speculative-num-steps 5 \
    --speculative-eagle-topk 24 --speculative-num-draft-tokens 128 \
    --cuda-graph-max-bs 1 --mem-fraction 0.7 --dtype bfloat16
    

python3 -m sglang.launch_server --model meta-llama/Meta-Llama-3-8B-Instruct  --speculative-algo EAGLE \
    --speculative-draft lmzheng/sglang-EAGLE-LLaMA3-Instruct-8B --speculative-num-steps 5 \
    --speculative-eagle-topk 1024 --speculative-num-draft-tokens 128 \
    --cuda-graph-max-bs 1 --mem-fraction 0.7 --dtype bfloat16

python3 -m sglang.launch_server --model meta-llama/Meta-Llama-3-8B-Instruct  --speculative-algo EAGLE \
    --speculative-draft lmzheng/sglang-EAGLE-LLaMA3-Instruct-8B --speculative-num-steps 5 \
    --speculative-eagle-topk 8 --speculative-num-draft-tokens 1024 \
    --cuda-graph-max-bs 1 --mem-fraction 0.7 --dtype bfloat16

python3 -m sglang.launch_server --model meta-llama/Meta-Llama-3-8B-Instruct  --speculative-algo EAGLE --speculative-draft lmzheng/sglang-EAGLE-LLaMA3-Instruct-8B --speculative-num-steps 5 --speculative-eagle-topk 8 --speculative-num-draft-tokens 64 --mem-fraction 0.7 --disable-radix --attention-backend triton --cuda-graph-max-bs 1 --dtype bfloat16


python3 -m sglang.launch_server --model meta-llama/Llama-2-7b-chat-hf  --speculative-algo EAGLE --speculative-draft lmzheng/sglang-EAGLE-llama2-chat-7B --speculative-num-steps 5 --speculative-eagle-topk 4 --speculative-num-draft-tokens 32 --mem-fraction 0.7 --cuda-graph-max-bs 1 --dtype float16

python3 -m sglang.launch_server --model meta-llama/Llama-2-7b-chat-hf  --speculative-algo EAGLE --speculative-draft lmzheng/sglang-EAGLE-llama2-chat-7B --speculative-num-steps 5 --speculative-eagle-topk 16 --speculative-num-draft-tokens 128 --mem-fraction 0.7 --cuda-graph-max-bs 1 --dtype float16


python3 -m sglang.launch_server --model meta-llama/Llama-2-7b-chat-hf  --speculative-algo EAGLE --speculative-draft lmzheng/sglang-EAGLE-llama2-chat-7B --speculative-num-steps 5 --speculative-eagle-topk 4 --speculative-num-draft-tokens 64 --mem-fraction 0.7 --cuda-graph-max-bs 1 --dtype float16

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
