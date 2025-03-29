from datasets import load_dataset

# dataset = load_dataset("gsm8k", "main")
# dataset.save_to_disk("benchmarks/gsm8k")

# dataset = load_dataset("HuggingFaceH4/MATH-500")
# print("\n=== Sample Structure ===")
# print("\nKeys:", dataset['test'][0].keys())
# print("\nSample content:")
# for key, value in dataset['test'][0].items():
#     print(f"\n{key}:")
#     print(value)

# dataset.save_to_disk("benchmarks/MATH-500")

# dataset = load_dataset("qwedsacf/competition_math")
# dataset.save_to_disk("benchmarks/competition_math")

# dataset = load_dataset("zzy1123/MATH_train_test_split")
# dataset.save_to_disk("benchmarks/MATH_train_test_split")

dataset = load_dataset("guanning/math500-verification-64ans")
dataset.save_to_disk("benchmarks/math500-verification-64ans")