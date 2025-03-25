# Transfer the verl saved checkpoints into huggingface version
# convenient for upload or evaluation

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
from glob import glob
from collections import defaultdict
import os
from torch.distributed._tensor import DTensor
from torch.distributed._tensor.placement_types import DTensorSpec
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed._shard.sharded_tensor import Shard
from torch.serialization import add_safe_globals

### Params
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
actor_dir = 'checkpoints/generator-verl/qwen2.5-gsm8k-grpo-0.5b/global_step_870/actor'
world_size = 2

def is_fsdp_tensor(tensor):
    return hasattr(tensor, '_local_metadata')

def load_state_dict(fsdp_checkpoint_path, world_size):
    safe_classes = [DTensor, DTensorSpec, DeviceMesh, Shard]
    for cls in safe_classes:
        torch.serialization.add_safe_globals([cls])
    
    state_dict = defaultdict(list)
    for rank in range(world_size):
        filepath = f"{fsdp_checkpoint_path}/model_world_size_{world_size}_rank_{rank}.pt"
        print('loading', filepath)
        this_state_dict = torch.load(filepath, weights_only=False)

        for key, value in this_state_dict.items():
            if isinstance(value, DTensor):
                value = value.to_local()
            if hasattr(value, 'tensor'):
                value = value.tensor
            if hasattr(value, '_local_metadata'):
                value = value.detach()
            value = torch.tensor(value.cpu().numpy())
            state_dict[key].append(value)

    final_state_dict = {}
    for key in state_dict:
        try:
            final_state_dict[key] = torch.cat(state_dict[key], dim=0)
        except Exception as e:
            print(f"Error concatenating key {key}: {e}")
            print(f"Shapes: {[t.shape for t in state_dict[key]]}")
            raise
    
    return final_state_dict, True

def main():
    output_dir = f'{actor_dir}/huggingface'
    fsdp_checkpoint_path = actor_dir
    huggingface_model_path = output_dir
    
    state_dict, is_fsdp = load_state_dict(fsdp_checkpoint_path, world_size)

    config = AutoConfig.from_pretrained(huggingface_model_path)
    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(state_dict)

    model.save_pretrained(output_dir, max_shard_size="10GB")

    tokenizer = AutoTokenizer.from_pretrained(huggingface_model_path)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()