### How to Run

We need `CUDA 12.1` to run the code.

First create a conda environment

```bash
conda create -n llm python=3.10
conda activate llm
```

Install core dependencies

```bash
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip3 install flash-attn --no-build-isolation
pip3 install -e .
```

Install other dependencies

```bash
pip3 install "ray[default]" debugpy
pip3 install omegaconf latex2sympy2 antlr4-python3-runtime==4.9.3 hydra-core
pip3 install word2number
```

### How to Reproduce Evaluation Results 

Qwen2-Math-7B:

```bash
# About 81%
python evaluate_generator.py --model_path="./models/Qwen2-Math-7B" --dataset="./benchmarks/gsm8k" --tok_limit=4096 --split=test --test_n=1 --template="templates/Qwen_gsm8k_8shot.txt" --post_truncate
```

Qwen2.5-Math-1.5B:

```bash
# About 77%
python evaluate_generator.py --model_path="./models/Qwen2.5-Math-1.5B" --dataset="./benchmarks/gsm8k" --tok_limit=4096 --split=test --test_n=1 --template="templates/Qwen_gsm8k_8shot.txt" --post_truncate
```

Qwen2.5-Math-1.5B-Instruct:
```bash
# About 85%
python evaluate_generator.py --model_path="models/Qwen2.5-Math-1.5B-Instruct" --dataset="./benchmarks/gsm8k" --tok_limit=4096 --split=test --test_n=1 --template="templates/Qwen_gsm8k_CoT_0shot.txt" --post_truncate
```

Qwen2-1.5B-Instruct:

```bash
# About 62%
python evaluate_generator.py --model_path="models/Qwen2-1.5B-Instruct" --dataset="./benchmarks/gsm8k" --tok_limit=4096 --split=test --test_n=1 --template="templates/Qwen_gsm8k_0shot.txt"
```

Qwen2-0.5B-Instruct:

```bash
# About 39%
python evaluate_generator.py --model_path="models/Qwen2-0.5B-Instruct" --dataset="./benchmarks/gsm8k" --tok_limit=4096 --split=test --test_n=1 --template="templates/Qwen_gsm8k_4shot.txt" --post_truncate
```

### How to Train Generator

Download training data:

```bash
python benchmarks/download_benchmark.py
```

Download base model:

```bash
python models/download_model.py
```

Training `Qwen/Qwen2-0.5B-Instruct` on `gsm8k` with GRPO:

```bash
bash scripts/qwen2.5-0.5b_gsm8k_grpo.sh
```

### How to Train Verifier

TODO


