# Evaluate LLM's ability as a Verifier

import os, time
import json
from vllm import LLM, SamplingParams
from datasets import load_from_disk, load_dataset
from utils import DATASET_KEYS, RESPONSE_EXTRACTOR, RESPONSE_COMPARATOR
import pandas as pd
import argparse
import numpy as np
from tqdm import tqdm
from utils.utils import extract_model_shortname
from utils.parser import extract_yes_no_answer
os.makedirs('evaluations', exist_ok=True)

# Examples 
'''
CUDA_VISIBLE_DEVICES=1 python evaluate_verifier.py --model_path="./models/Qwen2.5-1.5B-Instruct" --dataset="./benchmarks/math500-verification" --tok_limit=8192 --split=train --test_n=1 --template="templates/verifier4.txt" --verification_type="yes_no"
'''

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-Math-1.5B')
parser.add_argument('--dataset', type=str, default='guanning/math500-verification')
parser.add_argument('--tok_limit', type=int, default=4096)
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--temperature', type=float, default=None)
parser.add_argument('--template', type=str, default=None)
parser.add_argument('--test_n', type=int, default=None)
parser.add_argument('--verification_type', type=str, default='scalar', choices=['scalar', 'yes_no'])

args = parser.parse_args()
os.environ['TOKENIZERS_PARALLELISM'] = "false"

dataset_name = args.dataset
model_path = args.model_path
tok_limit = args.tok_limit
split = args.split
dataset_name = args.dataset
dataset_short_name = dataset_name.split('/')[-1]
results = {}
template = args.template
verification_type = args.verification_type
template_short_name = template.split('/')[-1].split('.')[0] if template is not None else ''

if template is not None:
    with open(template, 'r', encoding='utf-8') as f:
        template = f.read()

print("Dataset:", dataset_short_name, "\nModel:", model_path)

QUESTION_KEY = DATASET_KEYS[dataset_short_name]["question"]
ANSWER_KEY = DATASET_KEYS[dataset_short_name]["answer"]
extract_answer = RESPONSE_EXTRACTOR[dataset_short_name] if verification_type == 'scalar' else extract_yes_no_answer
eq = RESPONSE_COMPARATOR[dataset_short_name]

if dataset_short_name == 'math500-verification':
    dataset = load_from_disk(dataset_name)
    TEST_N = 1
    MAX_TOKENS = tok_limit
    TEST_TEMPERATURE = 0.0
    MAX_TEST_SAMPLES = 200
else:
    pass # TODO: add other datasets

print("Available splits in dataset:", dataset.keys()) 
print("Available keys in dataset:", dataset[split].column_names)

# override the default values
if args.temperature is not None:
    TEST_TEMPERATURE = float(args.temperature)
if args.test_n is not None:
    TEST_N = int(args.test_n)

def get_scores(ds, outputs, tokenizer_encode, save_file_name=None):
    results = []
    tot_tokens = 0
    tot_pass_rate = 0
    for input, output in tqdm(zip(ds, outputs), total=len(ds), desc='Analysed responses'):
        truncated_responses = [resp.text for resp in output.outputs]
        prediction = [
            extract_answer(truncated_resp)
            for truncated_resp in truncated_responses
        ]
        result = {
            QUESTION_KEY: input[QUESTION_KEY] if not template else template.replace('<question>', input['original_problem']).replace('<response>', input['original_response']),
            ANSWER_KEY: input[ANSWER_KEY],
            "verification_thoughts": truncated_responses,  
            "verification_prediction": prediction,
            "tokens": sum([len(tokenizer_encode(truncated_resp)) for truncated_resp in truncated_responses]) / len(truncated_responses),
            "verification_accuracy": [eq(str(input[ANSWER_KEY]), str(pred)) for pred in prediction],
        }
        results.append(result)
        tot_tokens += result['tokens']
        tot_pass_rate += sum(result['verification_accuracy']) / len(result['verification_accuracy'])
        
    if save_file_name is not None:
        with open(save_file_name, 'w') as f:
            json.dump(results, f, indent=4)
    
    return {
        'avg_tokens': tot_tokens / len(results),
        'avg_verification_accuracy': tot_pass_rate / len(results),
    } # TODO: add various metrics when required

def evaluate_model():
    test_prompts = []
    model = LLM(model_path, tokenizer=model_path, gpu_memory_utilization=0.9, 
                tensor_parallel_size=1, max_model_len = MAX_TOKENS, swap_space=80)    
    
    test_ds = dataset[split].shuffle(seed=0).select(range(min(MAX_TEST_SAMPLES, len(dataset[split]))))
    
    for x in test_ds:
        prompt = x[QUESTION_KEY] if not template else template.replace('<question>', x['original_problem']).replace('<response>', x['original_response'])
        prompt_tokens = model.llm_engine.tokenizer.tokenizer.encode(prompt)
        test_prompts.append(prompt_tokens)
    
    sampling_params = SamplingParams(
        temperature=TEST_TEMPERATURE,
        max_tokens=MAX_TOKENS,
        n=TEST_N
    )

    sampling_params.stop_token_ids = [model.llm_engine.tokenizer.tokenizer.eos_token_id]
    print("Generating test outputs...")
    print(model.llm_engine.tokenizer.tokenizer.decode(test_prompts[0], skip_special_tokens=False))
    start_time = time.time()
    test_outputs = model.generate(prompt_token_ids=test_prompts, sampling_params=sampling_params, use_tqdm=True)
    test_scores = get_scores(test_ds, 
                             test_outputs, 
                             model.llm_engine.tokenizer.tokenizer.encode,
                             f"evaluations/verifier_outputs_{dataset_short_name}_{extract_model_shortname(model_path)}_{template_short_name}_{TEST_TEMPERATURE}_{tok_limit}.json")
    print("Test:", test_scores)
    end_time = time.time()
    time_taken = end_time - start_time
    print("Time taken:", time_taken)

    return {'test': test_scores, 'time_taken': time_taken}

print("Found model_path:", model_path)
print("This is not a checkpoint, will evaluate directly...")
scores = evaluate_model()
results[model_path] = scores

result_file = f'evaluations/verifier_results_{dataset_short_name}_{extract_model_shortname(model_path)}_{template_short_name}_{TEST_TEMPERATURE}_{tok_limit}.json'
with open(result_file, 'w') as f:
    print('Saving results to', result_file)
    json.dump(results, f, indent=4)
