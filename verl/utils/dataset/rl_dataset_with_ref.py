from verl.utils.dataset.rl_dataset import RLHFDataset
from typing import Union, List, Optional
from transformers import PreTrainedTokenizer, ProcessorMixin

class RLHFDatasetWithRef(RLHFDataset):

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 processor: Optional[ProcessorMixin] = None,
                 prompt_key='prompt',
                 reference_key='gold',
                 **kwargs):
        self.reference_key = reference_key
        
        super().__init__(
            parquet_files=parquet_files,
            tokenizer=tokenizer,
            processor=processor,
            prompt_key=prompt_key,
            **kwargs
        )

    def __getitem__(self, item):

        row_dict = super().__getitem__(item)

        reference = self.dataframe.iloc[item][self.reference_key]
        row_dict['reference'] = reference
        
        return row_dict