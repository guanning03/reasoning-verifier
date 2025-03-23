# Transfer Huggingface-type Dataset to Parquet
# Required by verl training & evaluation (RLHFDataset)

import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse

# Example:
'''
python data-transfer_verl.py --dataset_path="./benchmarks/math500-verification" --wo_test
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='./benchmarks/math500-verification')
    parser.add_argument('--wo_train', action='store_true')
    parser.add_argument('--wo_test', action='store_true')

    args = parser.parse_args()

    dataset_path = args.dataset_path

    dataset = datasets.load_from_disk(dataset_path)

    train_dataset = test_dataset = None
    if not args.wo_train:
        train_dataset = dataset['train']
    if not args.wo_test:
        test_dataset = dataset['test']

    instruction_following = "Let's think step by step and output the final answer after \"####\"."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            
            data = {
                "data_source": dataset_path.split('/')[-1],
                "prompt": [{
                    "role": "user",
                    "content": example.pop('content_to_verify'),
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": example.pop('verification')
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'answer': example.pop('original_solution'),
                    "question": example.pop('original_problem'),
                    "response": example.pop('original_response'),
                    "gold": example.pop('original_gold'),
                    "prediction": example.pop('original_prediction'),
                }
            }
            return data

        return process_fn

    if not args.wo_train:
        train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
        train_dataset.to_parquet(os.path.join(dataset_path, 'train.parquet'))
    if not args.wo_test:
        test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
        test_dataset.to_parquet(os.path.join(dataset_path, 'test.parquet'))


