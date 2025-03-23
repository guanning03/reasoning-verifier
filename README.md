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



