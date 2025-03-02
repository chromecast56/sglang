"""
python test_llamalora.py --model_name /data/franklin/checkpoints/phoenix/phoenix-pretrain-1layer-5epochs-lora-target/checkpoint-93225 --adapter_path /data/franklin/checkpoints/phoenix/phoenix-pretrain-1layer-5epochs-lora-target/adapter_config.json --output_path /data/jamesliu/sglang/merged_1layer_ref/

"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft.mapping import get_peft_config
from peft import PeftModel
from peft.config import PeftConfig
from peft.peft_model import PeftModelForCausalLM

import copy


def main():
    parser = argparse.ArgumentParser(description="Merge a model with a LoRA adapter")
    parser.add_argument("--model_name", type=str, required=True, help="Path to the base model")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the adapter config")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the merged model")
    parser.add_argument("--device", type=str, default="auto", help="Device to load the model on")
    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Loading base model from {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    base_weights = model.model.layers[0].self_attn.q_proj.weight.data.clone()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    print(f"Loading adapter from {args.adapter_path}")
    adapter_config = get_peft_config(PeftConfig.from_json_file(args.adapter_path))
    print(adapter_config)
    adapted_target_model = PeftModelForCausalLM(copy.deepcopy(model), adapter_config)

    state_dict = adapted_target_model.state_dict()
    # print(state_dict.keys())
    print(state_dict["base_model.model.model.layers.30.mlp.gate_proj.lora_B.default.weight"])

    target_model_weights_only = {k: v for k, v in state_dict.items() if not k.startswith("phoenix.")}

    # print(target_model_weights_only.keys())
    print(adapted_target_model.model.model.layers[0].self_attn.q_proj.lora_B.default.weight.data)
    adapted_target_model.load_state_dict(target_model_weights_only)

    print(adapted_target_model.model.model.layers[0].self_attn.q_proj.lora_B.default.weight.data)

    # print("Merging adapter weights into base model")
    adapted_target_model.merge_and_unload()

    # Add some verification to see if weights changed
    
    merged_weights = adapted_target_model.model.model.layers[0].self_attn.q_proj.weight.data
    print(f"Weight difference: {torch.sum(torch.abs(merged_weights - base_weights))}")
    
    print(f"Saving merged model to {args.output_path}")
    adapted_target_model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    
    print("Done!")



if __name__ == "__main__":
    main()