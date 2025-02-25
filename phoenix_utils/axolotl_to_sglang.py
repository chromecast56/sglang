"""
Script to split a merged HF checkpoint containing a base model and a phoenix head
into two separate models that are Llama-style.

Usage:
    python axolotl_to_sglang.py \
         --merged_model_dir /data/jamesliu/models/models--togethercomputer--phoenix-1layer-baseline/snapshots/8261c2483ce3382c5c1d5d14d77ca42386bc5a75/checkpoint-93225 \
         --base_model_dir /data/jamesliu/sglang/phoenix_1layer_baseline/base_model \
         --phoenix_model_dir /data/jamesliu/sglang/phoenix_1layer_baseline/phoenix_model --phoenix_num_layers 1

For LoRA:
    python axolotl_to_sglang.py \
         --merged_model_dir /data/franklin/checkpoints/phoenix/phoenix-pretrain-1layer-5epochs-lora-target/checkpoint-93225 \
         --base_model_dir /data/jamesliu/sglang/phoenix_1layer_lora/base_model \
         --phoenix_model_dir /data/jamesliu/sglang/phoenix_1layer_lora/phoenix_model --phoenix_num_layers 1 --lora_rank 64

python axolotl_to_sglang.py \
         --merged_model_dir /data/franklin/checkpoints/phoenix/phoenix-pretrain-2layer-5epochs-lora-target/checkpoint-93225 \
         --base_model_dir /data/jamesliu/sglang/phoenix_2layer_lora/base_model \
         --phoenix_model_dir /data/jamesliu/sglang/phoenix_2layer_lora/phoenix_model --phoenix_num_layers 2 --lora_rank 64

         
For example, if you trained togethercomputer/phoenix-1layer-baseline, point
--merged_model_dir to the checkpoint (e.g. checkpoint-93225).
"""

import os
import glob
import argparse

import torch
import torch.nn as nn
from transformers import AutoConfig, LlamaForCausalLM, PretrainedConfig

def get_original_modules(layer):
    return [
        layer.self_attn.q_proj,
        layer.self_attn.k_proj,
        layer.self_attn.v_proj,
        layer.self_attn.o_proj,
        layer.mlp.gate_proj,
        layer.mlp.up_proj,
        layer.mlp.down_proj,
    ]



class LoRAModule(nn.Module):
    def __init__(self, output_size, input_size, lora_rank):
        super().__init__()

        self.base_layer = nn.Linear(input_size, output_size, bias=False)
        self.lora_A = nn.Linear(input_size, lora_rank, bias=False)
        self.lora_B = nn.Linear(lora_rank, output_size, bias=False)

class LlamaForCausalLMLoRA(LlamaForCausalLM):
    def __init__(
        self,
        config,
        lora_rank,
    ) -> None:
        super().__init__(config)

        for layer in self.model.layers:
            for module in get_original_modules(layer):
                del module

            layer.self_attn.q_proj = LoRAModule(config.head_dim*config.num_attention_heads, config.hidden_size, lora_rank)
            layer.self_attn.k_proj = LoRAModule(config.head_dim*config.num_key_value_heads, config.hidden_size, lora_rank)
            layer.self_attn.v_proj = LoRAModule(config.head_dim*config.num_key_value_heads, config.hidden_size, lora_rank)
            layer.self_attn.o_proj = LoRAModule(config.hidden_size, config.head_dim*config.num_attention_heads, lora_rank)
            layer.mlp.gate_proj = LoRAModule(config.intermediate_size, config.hidden_size, lora_rank)
            layer.mlp.up_proj = LoRAModule(config.intermediate_size, config.hidden_size, lora_rank)
            layer.mlp.down_proj = LoRAModule(config.hidden_size, config.intermediate_size, lora_rank)

# We also import (or define) the phoenix variant class.
# Here we simply subclass LlamaForCausalLM so that we can give it a new architecture name.
class LlamaForCausalLMPhoenix(LlamaForCausalLM):
    def __init__(
        self,
        config,
    ) -> None:
        super().__init__(config)

        self.model.fc = nn.Linear(config.hidden_size*2, config.hidden_size, bias=True)


def extract_state_dicts(merged_dir):
    """
    Load all safetensors files in merged_dir and merge them into one state_dict.
    Then filter out the keys for the base model (prefixed with "model.model.")
    and for the phoenix part (prefixed with "phoenix_head.").
    
    Returns:
       base_sd (dict): state dict for base model, with the prefix removed.
       phoenix_sd (dict): state dict for phoenix model (prefix removed).
    """
    from safetensors.torch import load_file

    merged_sd = {}
    # The safetensors files can be sharded, e.g. model-00001-of-00004.safetensors…
    files = glob.glob(os.path.join(merged_dir, "*.safetensors"))
    if len(files) == 0:
        raise ValueError(f"No safetensors files found in directory {merged_dir}")
    for f in files:
        print(f"Loading {f} …")
        sd = load_file(f)
        merged_sd.update(sd)

    base_sd = {}
    phoenix_sd = {}
    for key, value in merged_sd.items():
        # Our training script saved base model state dict keys with a prefix "model.model."
        # and the phoenix model keys with prefix "phoenix_head."


        # lora
        if key.startswith("model.base_model.model."):
            new_key = key[len("model.base_model.model."):]
            # Remove ".default" from keys for LoRA modules (lora_A or lora_B)
            import re
            new_key = re.sub(r"(lora_[AB])\.default", r"\1", new_key)
            base_sd[new_key] = value

            print(key, new_key)

        # nonlora
        elif key.startswith("model.model."):
            # Remove "model.model." so that it matches the expected keys in LlamaForCausalLM
            new_key = key[len("model."):]
            base_sd[new_key] = value
            # print(key, new_key)
        elif key.startswith("model."):
            new_key = key[len("model."):]
            base_sd[new_key] = value


        elif key.startswith("phoenix_head.fc"):
            new_key = f"model.{key[len('phoenix_head.'):]}"
            phoenix_sd[new_key] = value
            # print(key, new_key)

        elif key.startswith("phoenix_head.model."):
            # Remove prefix "phoenix_head.".
            new_key = key[len("phoenix_head.model."):]
            # print(key, new_key)
            phoenix_sd[new_key] = value

            # if key.startswith("phoenix_head.model.lm_head.weight"):
                # print(value)

        elif key.startswith("phoenix_head."):
            # Remove prefix "phoenix_head.".
            new_key = key[len("phoenix_head."):]
            phoenix_sd[new_key] = value
            # print(key, new_key)
        else:
            # (Optional) warn or ignore any unexpected keys.
            print(f"Warning: key '{key}' does not match expected prefixes; ignoring.")
    return base_sd, phoenix_sd


def save_separated_models(merged_dir, base_model_dir, phoenix_model_dir, phoenix_num_layers, lora_rank=None):
    """
    Loads the merged checkpoint from merged_dir, splits the state dict
    into base and phoenix portions, creates two separate HF models, and saves them.
    """
    # load config from merged directory (i.e. the original config.json)
    config = AutoConfig.from_pretrained(merged_dir)

    # print(config)

    dtype = config.torch_dtype if isinstance(config.torch_dtype, torch.dtype) else getattr(torch, config.torch_dtype)

    
    # Extract our two state dicts.
    base_sd, phoenix_sd = extract_state_dicts(merged_dir)

    # print(phoenix_sd.keys())
    
    # ----- Create and save the Base model -----
    print("Loading base model …")
    # Instantiate a standard llama model with the given config.

    config._name_or_path = base_model_dir
    if lora_rank is None:
        base_model = LlamaForCausalLM(config).to(dtype)
    else:
        config.architectures = ["LlamaForCausalLMLoRA"]
        config.model_type = "llama"
        config.lora_rank = lora_rank
        base_model = LlamaForCausalLMLoRA(config, lora_rank).to(dtype)

    missing_keys, unexpected_keys = base_model.load_state_dict(base_sd, strict=False)

    if missing_keys:
        print("Base model missing keys:", missing_keys)
    if unexpected_keys:
        print("Base model unexpected keys:", unexpected_keys)
    print("Saving base model to:", base_model_dir)
    base_model.save_pretrained(base_model_dir)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(merged_dir)
    tokenizer.save_pretrained(base_model_dir)

    # ----- Create and save the Phoenix model -----
    print("Loading phoenix model …")




    # Change the architectures field to use our phoenix class.
    config._name_or_path = phoenix_model_dir
    config.architectures = ["LlamaForCausalLMPhoenix"]
    config.num_hidden_layers = phoenix_num_layers
    config.model_type = "llama"
    
    # Instantiate the phoenix model.
    phoenix_model = LlamaForCausalLMPhoenix(config).to(dtype)
    missing_keys, unexpected_keys = phoenix_model.load_state_dict(phoenix_sd, strict=False)
    if missing_keys:
        print("Phoenix model missing keys:", missing_keys)
    if unexpected_keys:
        print("Phoenix model unexpected keys:", unexpected_keys)
    
    
    print("Saving phoenix model to:", phoenix_model_dir)
    phoenix_model.save_pretrained(phoenix_model_dir)


def main():
    parser = argparse.ArgumentParser(description="Split merged phoenix-base HF model into separate models.")
    parser.add_argument("--merged_model_dir", type=str, required=True,
                        help="Directory containing the merged checkpoint (config.json and safetensors files).")
    parser.add_argument("--base_model_dir", type=str, required=True,
                        help="Output directory for the base Llama model.")
    parser.add_argument("--phoenix_model_dir", type=str, required=True,
                        help="Output directory for the phoenix model.")

    parser.add_argument("--phoenix_num_layers", type=int, required=True,
                        help="Number of layers in the phoenix model.")

    parser.add_argument("--lora_rank", type=int, default=None,
                        help="The rank of the LoRA modules.")

    args = parser.parse_args()

    # Create output directories if needed.
    os.makedirs(args.base_model_dir, exist_ok=True)
    os.makedirs(args.phoenix_model_dir, exist_ok=True)
    
    save_separated_models(args.merged_model_dir, args.base_model_dir, args.phoenix_model_dir, args.phoenix_num_layers, args.lora_rank)
    print("Separation complete.")


if __name__ == "__main__":
    main()