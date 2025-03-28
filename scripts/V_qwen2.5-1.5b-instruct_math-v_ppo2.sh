set -x
mkdir -p logs/
CURRENT_TIME=$(date "+%Y%m%d_%H%M%S")
echo "Job started on `hostname` at `date`"

# ray stop

export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export WANDB_MODE='online'

export VLLM_ATTENTION_BACKEND=XFORMERS

math_verification_train_path=./benchmarks/math-verification/train.parquet
math_verification_test_path=./benchmarks/math-verification/test.parquet

train_files="['$math_verification_train_path']"
test_files="['$math_verification_test_path']"

python3 -m verl.trainer.main_ppo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.disable_chat_template=True \
    actor_rollout_ref.model.path=models/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.optim.lr=1e-5 \
    critic.model.path=models/Qwen2.5-1.5B-Instruct \
    critic.ppo_micro_batch_size_per_gpu=4 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verifier-verl' \
    trainer.experiment_name='qwen2.5-1.5b-instruct_math-v_ppo2' \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=20 \
    trainer.max_ckpt_to_keep=20 \
    trainer.total_epochs=80 $@ 2>&1 | tee logs/${CURRENT_TIME}.log