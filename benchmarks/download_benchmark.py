from datasets import load_dataset

dataset = load_dataset("gsm8k", "main")
dataset.save_to_disk("benchmarks/gsm8k")

dataset = load_dataset("HuggingFaceH4/MATH-500")
dataset.save_to_disk("benchmarks/MATH-500")

dataset = load_dataset("qwedsacf/competition_math")
dataset.save_to_disk("benchmarks/competition_math")

