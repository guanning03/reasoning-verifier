from huggingface_hub import HfApi

# 初始化 Huggingface API
api = HfApi()

# 创建新的模型仓库
repo_id = "guanning/Verifier-Qwen2.5-1.5B-Instruct-PPO-600"
api.create_repo(repo_id, private=False)

# 上传模型文件
api.upload_folder(
    folder_path="checkpoints/verifier-verl/qwen2.5-1.5b-instruct_math-v_ppo/global_step_600/actor/huggingface",
    repo_id=repo_id,
    repo_type="model"
)