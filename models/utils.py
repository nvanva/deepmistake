from transformers import BertConfig
from transformers import XLMRobertaConfig, RobertaConfig
from torch.utils.data import DataLoader, TensorDataset
from .xlmr_qa import XLMRQAModel
from .xlmr import XLMRModel
import torch
import os
import json
from collections import Counter
import pandas as pd
from glob import glob
import numpy as np
from collections import namedtuple

Example = namedtuple('Example',
                     ['docId', 'pos', 'text_1', 'text_2', 'label', 'start_1', 'end_1', 'start_2', 'end_2', 'score',
                      'lemma', 'grp'])


class DataProcessor:

    def get_examples(self, source_dir):
        data_files = glob(os.path.join(source_dir, '*.data'))
        examples = []
        for file in data_files:
            data = json.load(open(file, encoding='utf-8'))
            gold_file = f'{file[:-5]}.gold'
            gold_labels = json.load(open(gold_file, encoding='utf-8')) if os.path.exists(gold_file) else [{'tag': 'F'}] * len(data)
            for ex, lab in zip(data, gold_labels):
                pos = ex['pos'].lower()
                label = lab['tag']
                if 'score' in lab:
                    score = lab['score']
                else:
                    score = -1.0
                if 'grp' in ex:
                    grp = ex['grp']
                else:
                    grp = '?'
                examples.append(Example(ex['id'], pos, ex['sentence1'], ex['sentence2'], label, ex['start1'], ex['end1'], ex['start2'], ex['end2'], score, ex['lemma'], grp))
        return examples

def get_qa_dataloader_and_tensors(
        features: list,
        batch_size: int
):
    input_ids = torch.tensor(
        [f.input_ids for f in features],
        dtype=torch.long
    )
    input_mask = torch.tensor(
        [f.input_mask for f in features],
        dtype=torch.long
    )
    token_type_ids = torch.tensor(
        [f.token_type_ids for f in features],
        dtype=torch.long
    )
    syn_labels = torch.tensor(
        [f.syn_labels for f in features],
        dtype=torch.long
    )
    pos_label = torch.tensor(
        [f.pos_label for f in features],
        dtype=torch.long
    )
    eval_data = TensorDataset(
        input_ids, input_mask, token_type_ids,
        syn_labels, pos_label
    )

    dataloader = DataLoader(eval_data, batch_size=batch_size)

    return dataloader, syn_labels, pos_label

def get_dataloader_and_tensors(
        features: list,
        batch_size: int
):
    input_ids = torch.tensor(
        [f.input_ids for f in features],
        dtype=torch.long
    )
    input_mask = torch.tensor(
        [f.input_mask for f in features],
        dtype=torch.long
    )
    token_type_ids = torch.tensor(
        [f.token_type_ids for f in features],
        dtype=torch.long
    )
    syn_labels = torch.tensor(
        [f.syn_label for f in features]
    )
    positions = torch.tensor(
        [f.positions for f in features],
        dtype=torch.long
    )
    eval_data = TensorDataset(
        input_ids, input_mask, token_type_ids,
        syn_labels, positions
    )

    dataloader = DataLoader(eval_data, batch_size=batch_size)

    return dataloader


qa_models = {
    # "bert-large-uncased": XLNetForQA,
    "roberta-base": XLMRQAModel,
    "roberta-large": XLMRQAModel,
    "xlm-roberta-base": XLMRQAModel,
    "xlm-roberta-large": XLMRQAModel
    # "xlnet-large-cased": BertForQA
}

models = {
    # "bert-large-uncased": XLNetForQA,
    "roberta-base": XLMRModel,
    "roberta-large": XLMRModel,
    "xlm-roberta-base": XLMRModel,
    "xlm-roberta-large": XLMRModel
    # "xlnet-large-cased": BertForQA
}

configs = {
    "bert-large-uncased": BertConfig,
    "roberta-large": RobertaConfig,
    "roberta-base": RobertaConfig,
    "xlm-roberta-base": XLMRobertaConfig,
    "xlm-roberta-large": XLMRobertaConfig
}
