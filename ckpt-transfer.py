# Transfer the verl saved checkpoints into huggingface version
# convenient for upload or evaluation

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
from glob import glob
from collections import defaultdict
import os

### Params
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
actor_dir = 'checkpoints/qwen2.5-gsm8k-ppo/qwen2.5-gsm8k-ppo-0.5b-0shot-0.001kl-256bs-512ml-256rl-1e-6lr-1e-5cr/global_step_435/critic'
world_size = 1

def is_fsdp_tensor(tensor):
    return hasattr(tensor, '_local_metadata')

def load_state_dict(fsdp_checkpoint_path, world_size):
    state_dict = defaultdict(list)
    for rank in range(world_size):
        filepath = f"{fsdp_checkpoint_path}/model_world_size_{world_size}_rank_{rank}.pt"
        print('loading', filepath)
        this_state_dict = torch.load(filepath, weights_only=True)

        first_tensor = next(iter(this_state_dict.values()))
        is_fsdp = is_fsdp_tensor(first_tensor)
        
        for key, value in this_state_dict.items():
            if is_fsdp:
                state_dict[key].append(value.to_local())
            else:
                state_dict[key].append(value)
    
    return state_dict, is_fsdp

def main():
    output_dir = f'{actor_dir}/huggingface'
    fsdp_checkpoint_path = actor_dir
    huggingface_model_path = output_dir
    
    state_dict, is_fsdp = load_state_dict(fsdp_checkpoint_path, world_size)

    for key in state_dict:
        state_dict[key] = torch.cat(state_dict[key], dim=0)

    config = AutoConfig.from_pretrained(huggingface_model_path)
    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(state_dict)

    model.save_pretrained(output_dir, max_shard_size="10GB")

    tokenizer = AutoTokenizer.from_pretrained(huggingface_model_path)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()