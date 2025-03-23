# 训练脚本

data.train_batch_size=256 ——每次explore用256个数据，相当于OpenRLHF的rollout_batch_size

actor_rollout_ref.actor.ppo_mini_batch_size=64 ——每次参数更新用64个数据

actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 ——每单个GPU上，每次参数更新用4个数据

actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 ——重要性采样的micro batch size

