# Creating Verifier Training Dataset
# From Generator Evaluation (json) to Huggingface (arrow)

import json
from datasets import Dataset
import os

def read_template(template_path):
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()

def create_content_to_verify(problem, response, template):
    content = template.replace('<question>', problem)
    content = content.replace('<response>', response)
    return content

def process_math_data(train_json_path, test_json_path, template_path):
    template = read_template(template_path)
    
    def load_data(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    train_data = load_data(train_json_path)
    test_data = load_data(test_json_path)
    
    def create_dataset_dict(data):
        dataset_dict = {
            'problem_idx': [], 'response_idx': [],
            'content_to_verify': [], 'original_problem': [],
            'original_response': [], 'verification': [],
            'original_prediction': [], 'original_gold': [],
            'original_solution': []
        }
        
        problem_idx = 0
        for item in data:
            problem = item['problem']
            response = item['responses']
            predictions = item['prediction']
            gold = item['gold']
            accuracy = item['accuracy']
            solution = item['solution']

            response_idx = 0 
            for idx, pred in enumerate(predictions):
                dataset_dict['problem_idx'].append(problem_idx)
                dataset_dict['response_idx'].append(response_idx)
                response_idx += 1
                dataset_dict['content_to_verify'].append(create_content_to_verify(problem, response[idx], template))
                dataset_dict['original_problem'].append(problem)
                dataset_dict['original_response'].append(response[idx])
                dataset_dict['original_prediction'].append(pred)
                dataset_dict['original_gold'].append(gold)
                dataset_dict['verification'].append(int(accuracy[idx]))
                dataset_dict['original_solution'].append(solution)
            problem_idx += 1
        return dataset_dict
    
    train_dataset_dict = create_dataset_dict(train_data)
    test_dataset_dict = create_dataset_dict(test_data)
    
    train_dataset = Dataset.from_dict(train_dataset_dict)
    test_dataset = Dataset.from_dict(test_dataset_dict)
    
    return train_dataset, test_dataset

def main():
    train_json_path = 'evaluations/outputs_MATH_train_test_split_Qwen2.5-Math-1.5B_test_500_Qwen_MATH_0shot_0.6_3072.json'
    test_json_path = 'evaluations/outputs_MATH_train_test_split_Qwen2.5-Math-1.5B_test_500_Qwen_MATH_0shot_0.6_3072.json'
    template_path = 'templates/verifier4.txt'
    huggingface_datarepo = 'guanning/math500-verification-64ans'
    
    train_dataset, test_dataset = process_math_data(train_json_path, test_json_path, template_path)

    print("\n=== Dataset Information ===")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    print("\nDataset features:", train_dataset.features)

    # train_dataset.push_to_hub(huggingface_datarepo, split="train", private=False)
    test_dataset.push_to_hub(huggingface_datarepo, split="test", private=False)

if __name__ == "__main__":
    main()
