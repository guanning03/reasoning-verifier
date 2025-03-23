# Transfer the verl saved checkpoints into huggingface version
# convenient for upload or evaluation

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
from glob import glob
from collections import defaultdict

actor_dir = 'checkpoints/qwen2.5-gsm8k-ppo/qwen2.5-gsm8k-ppo-0.5b-0shot-0.001kl-256bs-512ml-256rl-1e-6lr-1e-5cr/global_step_435/actor'
output_dir = f'{actor_dir}/huggingface_checkpoint'
world_size = 8

def main():
    fsdp_checkpoint_path = actor_dir
    huggingface_model_path = output_dir
    
    state_dict = defaultdict(list)

    # 使用全局定义的world_size
    for rank in range(world_size):
        filepath = f"{fsdp_checkpoint_path}/model_world_size_{world_size}_rank_{rank}.pt"
        print('loading', filepath)
        this_state_dict = torch.load(filepath)
        for key, value in this_state_dict.items():
            state_dict[key].append(value.to_local())

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