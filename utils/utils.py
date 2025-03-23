from utils.parser import extract_answer
from utils.grader import math_equal

DATASET_KEYS = {
    'gsm8k': {'question': 'question', 'answer': 'answer'},
    'competition_math': {'question': 'problem', 'answer': 'solution'},
    'converted_aime_dataset': {'question': 'problem', 'answer': 'solution'},
    'MATH500': {'question': 'problem', 'answer': 'solution'},
    'MATH-500': {'question': 'problem', 'answer': 'solution'},
    'compression_dataset': {'question': 'problem', 'answer': 'solution'},
    'AIME_2024': {'question': 'Problem', 'answer': 'Solution'},
    'AIME2025': {'question': 'question', 'answer': 'answer'},
    'math500-verification': {'question': 'content_to_verify', 'answer': 'verification'}
}

RESPONSE_EXTRACTOR = {
    'gsm8k': lambda x: extract_answer(x, data_name='gsm8k'),
    'competition_math': lambda x: extract_answer(x, data_name='math'),
    'MATH500': lambda x: extract_answer(x, data_name='math'),
    'MATH-500': lambda x: extract_answer(x, data_name='math'),
    'compression_dataset': lambda x: extract_answer(x, data_name='math'),
    'converted_aime_dataset': lambda x: extract_answer(x, data_name='math'),
    'AIME_2024': lambda x: extract_answer(x, data_name='math'),
    'AIME2025': lambda x: extract_answer(x, data_name='math'),
    'math500-verification': lambda x: extract_answer(x, data_name='math')
}

RESPONSE_COMPARATOR = {
    'gsm8k': lambda x, y: math_equal(x, y, timeout=True),
    'competition_math': lambda x, y: math_equal(x, y, timeout=True),
    'MATH500': lambda x, y: math_equal(x, y, timeout=True),
    'MATH-500': lambda x, y: math_equal(x, y, timeout=True),
    'compression_dataset': lambda x, y: math_equal(x, y, timeout=True),
    'converted_aime_dataset': lambda x, y: math_equal(x, y, timeout=True),
    'AIME_2024': lambda x, y: math_equal(x, y, timeout=True),
    'AIME2025': lambda x, y: math_equal(x, y, timeout=True),
    'math500-verification': lambda x, y: math_equal(x, y, timeout=True)
}

def extract_model_shortname(model_path):
    parts = model_path.split('/')
    for i in range(len(parts)-1, -1, -1):
        part = parts[i]
        if part in ['huggingface', 'actor'] or part.startswith('global_step_'):
            continue
        if i < len(parts)-1 and parts[i+1].startswith('global_step_'):
            return f"{part}_{parts[i+1]}"
        return part
    return parts[-1]