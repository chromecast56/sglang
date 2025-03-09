## Download Dataset

```sh
wget -O question.jsonl https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl
```

## Run benchmark

### Benchmark sglang
```
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000

six python3 -m sglang.launch_server --model meta-llama/Llama-3.1-8B-Instruct --speculative-algo EAGLE3 \
    --speculative-draft jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B --speculative-num-steps 5 \
    --speculative-eagle-topk 8 --speculative-num-draft-tokens 64 \
    --disable-cuda-graph --mem-fraction 0.7 --dtype float16 --port 30001

python3 -m sglang.launch_server --model meta-llama/Meta-Llama-3-8B-Instruct --speculative-algorithm EAGLE \
    --speculative-draft-model-path lmzheng/sglang-EAGLE-LLaMA3-Instruct-8B --speculative-num-steps 5 \
    --speculative-eagle-topk 8 --speculative-num-draft-tokens 64 --speculative-token-map thunlp/LLaMA3-Instruct-8B-FR-Spec/freq_32768.pt \
    --disable-cuda-graph --mem-fraction 0.7 --dtype float16 --port 30001


six python3 -m sglang.launch_server --model meta-llama/Llama-2-7b-chat-hf --speculative-algo EAGLE \
    --speculative-draft lmsys/sglang-EAGLE-llama2-chat-7B --speculative-num-steps 5 \
    --speculative-eagle-topk 8 --speculative-num-draft-tokens 64 \
    --disable-cuda-graph --mem-fraction 0.7 --dtype float16

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
