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
            'content_to_verify': [],
            'original_problem': [],
            'original_response': [],
            'verification': [],
            'original_prediction': [],
            'original_gold': [],
            'original_solution': []
        }
    }
    
    for item in data:
        problem = item['problem']
        response = item['responses']
        predictions = item['prediction']
        gold = item['gold']
        accuracy = item['accuracy']
        solution = item['solution']

        for idx, pred in enumerate(predictions):
            dataset_dict['train']['content_to_verify'].append(create_content_to_verify(problem, response[idx], template))
            dataset_dict['train']['original_problem'].append(problem)
            dataset_dict['train']['original_response'].append(response[idx])
            dataset_dict['train']['original_prediction'].append(pred)
            dataset_dict['train']['original_gold'].append(gold)
            dataset_dict['train']['verification'].append(int(accuracy[idx]))
            dataset_dict['train']['original_solution'].append(solution)
            
    dataset = Dataset.from_dict(dataset_dict['train'])
    return dataset

def main():
    json_path = 'evaluations/MATH500_responses.json'
    template_path = 'templates/verifier3.txt'

    dataset = process_math500_data(json_path, template_path)

    print("\n=== Dataset Information ===")
    print(f"Dataset size: {len(dataset)}")
    print("\nDataset features:", dataset.features)
    print("\nFirst example:")
    print(json.dumps(dataset[0], indent=2, ensure_ascii=False))

    dataset.save_to_disk('./benchmarks/math500-verification')

    dataset.push_to_hub(
        "guanning/math500-verification",  
        private=False 
    )

if __name__ == "__main__":
    main()