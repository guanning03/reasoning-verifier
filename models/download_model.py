from huggingface_hub import snapshot_download

# model_path = snapshot_download(
#     repo_id="Qwen/Qwen2.5-Math-1.5B",
#     local_dir="./models/Qwen2.5-Math-1.5B"  
# )

# model_path = snapshot_download(
#     repo_id="Qwen/Qwen2.5-7B-Instruct",
#     local_dir="./models/Qwen2.5-7B-Instruct"  
# )

# model_path = snapshot_download(
#     repo_id="Qwen/Qwen2.5-1.5B",
#     local_dir="./models/Qwen2.5-1.5B"  
# )

# model_path = snapshot_download(
#     repo_id="Qwen/Qwen2.5-1.5B",
#     local_dir="./models/Qwen2.5-1.5B"  
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
    repo_id="guanning/Verifier-Qwen2.5-1.5B-Instruct-PPO-600",
    local_dir="./models/Verifier-Qwen2.5-1.5B-Instruct-PPO-600"  
)

# model_path = snapshot_download(
#     repo_id="Qwen/Qwen2.5-1.5B-Instruct",
#     local_dir="./models/Qwen2.5-1.5B-Instruct"  
# )

# model_path = snapshot_download(
#     repo_id="Qwen/Qwen2-1.5B-Instruct",
#     local_dir="./models/Qwen2-1.5B-Instruct"  
# )

print('success')