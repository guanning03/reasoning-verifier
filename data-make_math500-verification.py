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

def process_math500_data(json_path, template_path):
    template = read_template(template_path)
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    dataset_dict = {
        'train': {
            'content_to_verify': [], 'original_problem': [],
            'original_response': [], 'verification': [],
            'original_prediction': [], 'original_gold': [],
            'original_solution': []
        },
        'test': { 
            'content_to_verify': [], 'original_problem': [],
            'original_response': [], 'verification': [],
            'original_prediction': [], 'original_gold': [],
            'original_solution': []
        }
    }
    
    for item_idx, item in enumerate(data):

        split = 'train' if item_idx < 450 else 'test'
        
        problem = item['problem']
        response = item['responses']
        predictions = item['prediction']
        gold = item['gold']
        accuracy = item['accuracy']
        solution = item['solution']

        for idx, pred in enumerate(predictions):
            dataset_dict[split]['content_to_verify'].append(create_content_to_verify(problem, response[idx], template))
            dataset_dict[split]['original_problem'].append(problem)
            dataset_dict[split]['original_response'].append(response[idx])
            dataset_dict[split]['original_prediction'].append(pred)
            dataset_dict[split]['original_gold'].append(gold)
            dataset_dict[split]['verification'].append(int(accuracy[idx]))
            dataset_dict[split]['original_solution'].append(solution)
            
    train_dataset = Dataset.from_dict(dataset_dict['train'])
    test_dataset = Dataset.from_dict(dataset_dict['test'])
    return train_dataset, test_dataset

def main():
    json_path = 'evaluations/MATH500_responses.json'
    template_path = 'templates/verifier4.txt'

    train_dataset, test_dataset = process_math500_data(json_path, template_path)

    print("\n=== Dataset Information ===")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    print("\nDataset features:", train_dataset.features)

    train_dataset.push_to_hub("guanning/math500-verification", split="train", private=False)
    test_dataset.push_to_hub("guanning/math500-verification", split="test", private=False)

if __name__ == "__main__":
    main()