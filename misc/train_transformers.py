import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
import torch

# 加载模型和分词器
model_name = "models/Qwen2.5-Math-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# 设置pad token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# 加载GSM8K数据集
dataset = load_dataset("gsm8k", "main")

# 定义数据处理函数
def preprocess_function(examples):
    questions = examples["question"]
    answers = examples["answer"]
    
    processed_inputs = []
    labels = []
    
    for question, answer in zip(questions, answers):
        # 构建完整提示
        full_text = f"Question: {question}\nLet's solve this step by step:\n{answer}"
        
        # Tokenize完整文本
        full_tokenized = tokenizer(full_text, truncation=True, max_length=4096)
        
        # Tokenize只有问题的部分
        question_part = f"Question: {question}\nLet's solve this step by step:"
        question_tokenized = tokenizer(question_part, add_special_tokens=False)
        
        # 计算问题部分的token长度
        question_length = len(question_tokenized["input_ids"])
        
        # 创建labels，对问题部分设置为-100（不计算损失）
        label_ids = [-100] * question_length + full_tokenized["input_ids"][question_length:]
        
        processed_inputs.append(full_tokenized["input_ids"])
        labels.append(label_ids)
    
    # 确保所有样本长度一致（padding）
    max_length = 4096
    padded_inputs = []
    padded_labels = []
    
    for input_ids, label_ids in zip(processed_inputs, labels):
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            label_ids = label_ids[:max_length]
        else:
            padding_length = max_length - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
            label_ids = label_ids + [-100] * padding_length
        
        padded_inputs.append(input_ids)
        padded_labels.append(label_ids)
    
    return {
        "input_ids": padded_inputs,
        "labels": padded_labels,
        "attention_mask": [[1 if id != tokenizer.pad_token_id else 0 for id in input_ids] for input_ids in padded_inputs]
    }

# 处理数据集
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# 训练参数
training_args = TrainingArguments(
    output_dir="./qwen-math-gsm8k-sft",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    bf16=True,
    report_to=[],
)

# 创建训练器（不需要DataCollator，因为我们手动处理了标签）
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

# 开始训练
trainer.train()

# 保存模型
trainer.save_model("./qwen-math-gsm8k-sft")