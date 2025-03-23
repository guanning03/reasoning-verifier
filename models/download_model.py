from huggingface_hub import snapshot_download

# model_path = snapshot_download(
#     repo_id="Qwen/Qwen2.5-Math-1.5B",
#     local_dir="./models/Qwen2.5-Math-1.5B"  
# )

# model_path = snapshot_download(
#     repo_id="Qwen/Qwen2.5-Math-1.5B-Instruct",
#     local_dir="./models/Qwen2.5-Math-1.5B-Instruct"  
# )

# model_path = snapshot_download(
#     repo_id="Qwen/Qwen2-Math-1.5B",
#     local_dir="./models/Qwen2-Math-1.5B"  
# )

model_path = snapshot_download(
    repo_id="Qwen/Qwen2.5-Math-0.5B",
    local_dir="./models/Qwen2.5-Math-0.5B"  
)

print('success')