import fire
import random
import os
import json
from glob import glob
import pandas as pd
import re


def prepare_dataset(dev_prop_to_train: int = 0.7, exclude_devs_from_split: str = 'dev.en-en', clean_all: bool = False):
    if clean_all:
        os.system('rm -rf data tmp rusemshift-*')
        return
    os.system('git clone https://github.com/SapienzaNLP/mcl-wic.git')
    os.system('unzip mcl-wic/SemEval-2021_MCL-WiC_trial.zip -d tmp')
    os.system('unzip mcl-wic/SemEval-2021_MCL-WiC_all-datasets.zip -d tmp')
    os.system('unzip mcl-wic/SemEval-2021_MCL-WiC_test-gold-data.zip -d tmp')
    os.system('rm -rf mcl-wic')
    train_dir, dev_dir, test_dir = 'data/train/', 'data/dev/', 'data/test/'
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(dev_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    for file in glob('tmp/MCL-WiC/training/*'):
        os.system(f'mv {file} {train_dir}')
    for file in glob('tmp/trial/*/*'):
        os.system(f'mv {file} {train_dir}')

    for file in glob('tmp/MCL-WiC/test/*/*'):
        os.system(f'mv {file} {test_dir}/')
    for file in glob('tmp/*.gold'):
        os.system(f'mv {file} {test_dir}/')
    # os.system(f'rm {test_dir}/*en-en*')
    # os.system(f'rm {test_dir}/*zh-zh*')
    # os.system(f'rm {test_dir}/*en-zh*')
    # os.system(f'rm {test_dir}/*en-ru*')

    for file in glob('tmp/MCL-WiC/dev/multilingual/*data'):
        file_name = file.split('/')[-1][:-5]
        if file_name in exclude_devs_from_split.split('+'):
            os.system(f'mv {file[:-5]}.gold {dev_dir}')
            os.system(f'mv {file} {dev_dir}')
            continue
        data = json.load(open(file, encoding='utf-8'))
        labs = json.load(open(file[:-5] + '.gold', encoding='utf-8'))
        unique_lemmas = sorted(set([ex['lemma'] for ex in data]))
        ids = list(range(len(unique_lemmas)))
        random.shuffle(ids)
        train_ids = ids[:int(dev_prop_to_train * len(ids))]
        train_lemmas = set([lemma for i, lemma in enumerate(unique_lemmas) if i in train_ids])
        train_data = [(ex, ex_lab) for ex, ex_lab in zip(data, labs) if ex['lemma'] in train_lemmas]
        dev_data = [(ex, ex_lab) for ex, ex_lab in zip(data, labs) if ex['lemma'] not in train_lemmas]

        json.dump([ex for (ex, _) in train_data], open(f'{train_dir}{file_name}.data', 'w', encoding='utf-8'), indent=4)
        json.dump([lab for (_, lab) in train_data], open(f'{train_dir}{file_name}.gold', 'w', encoding='utf-8'), indent=4)
        json.dump([ex for (ex, _) in dev_data], open(f'{dev_dir}{file_name}.data', 'w', encoding='utf-8'), indent=4)
        json.dump([lab for (_, lab) in dev_data], open(f'{dev_dir}{file_name}.gold', 'w', encoding='utf-8'), indent=4)

    os.system('rm -rf tmp')

    common_keys = ['id', 'lemma', 'pos', 'sentence1', 'sentence2']
    files = glob(f'data/*/*data') + glob(f'data/*/*/*data')
    print(files)
    for file in files:
        data = json.load(open(file, encoding='utf-8'))
        if 'start1' in data[0]:
            continue
        new_data = []
        print('modifying the following file:', file)
        for ex in data:
            new_ex = {k: ex[k] for k in common_keys}
            new_ex['start1'], new_ex['end1'] = [int(x) for x in ex['ranges1'].split('-')]
            new_ex['start2'], new_ex['end2'] = [int(x) for x in ex['ranges2'].split(',')[-1].split('-')]
            new_data.append(new_ex)
        json.dump(new_data, open(file, 'w', encoding='utf-8'), indent=4)

   
    os.system('git clone https://davletov-aa@bitbucket.org/nvanva/summer-lsc.git')
    os.makedirs('rusemshift-data', exist_ok=True)
    os.makedirs('rusemshift-tsvs', exist_ok=True)
    train_tsv = pd.read_csv('summer-lsc/datasets/rusemshift/train_1.tsv', sep='\t')
    train_tsv = train_tsv.append(pd.read_csv('summer-lsc/datasets/rusemshift/train_2.tsv', sep='\t'), ignore_index=True)
    dev_tsv_1 = pd.read_csv('summer-lsc/datasets/rusemshift/dev_1.tsv', sep='\t')
    dev_tsv_2 = pd.read_csv('summer-lsc/datasets/rusemshift/dev_2.tsv', sep='\t')
    assert len(set(train_tsv.word.unique()).intersection(
        set(dev_tsv_1.word.unique()))) == 0, 'non empty intersection of words in train and dev sets'
    assert len(set(train_tsv.word.unique()).intersection(
        set(dev_tsv_2.word.unique()))) == 0, 'non empty intersection of words in train and dev sets'

    train_tsv = train_tsv[
        (train_tsv.annotator1 != 0) & (train_tsv.annotator2 != 0) &
        (train_tsv.annotator3 != 0) & (train_tsv.annotator4 != 0) &
        (train_tsv.annotator5 != 0)
    ]
    train_tsv.to_csv('rusemshift-tsvs/train.tsv', sep='\t', index=False)
    dev_tsv_1 = dev_tsv_1[dev_tsv_1.group == 'COMPARE']
    dev_tsv_1 = dev_tsv_1[
        (dev_tsv_1.annotator1 != 0) & (dev_tsv_1.annotator2 != 0) &
        (dev_tsv_1.annotator3 != 0) & (dev_tsv_1.annotator4 != 0) &
        (dev_tsv_1.annotator5 != 0)
    ]
    dev_tsv_1.to_csv('rusemshift-tsvs/dev_1.tsv', sep='\t', index=False)
    dev_tsv_2 = dev_tsv_2[dev_tsv_2.group == 'COMPARE']
    dev_tsv_2 = dev_tsv_2[
        (dev_tsv_2.annotator1 != 0) & (dev_tsv_2.annotator2 != 0) &
        (dev_tsv_2.annotator3 != 0) & (dev_tsv_2.annotator4 != 0) &
        (dev_tsv_2.annotator5 != 0)
    ]
    dev_tsv_2.to_csv('rusemshift-tsvs/dev_2.tsv', sep='\t', index=False)

    for mode in ['mean', 'median']:
        data_dir = f'rusemshift-data/{mode}'
        os.makedirs(os.path.join(data_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'dev'), exist_ok=True)
        exs, labs = semshift2wic('rusemshift-tsvs/train.tsv', 'training.scd', mode=mode, split='train')
        json.dump(exs, open(os.path.join(data_dir, 'train', 'training.scd.data'), 'w'), indent=4)
        json.dump(labs, open(os.path.join(data_dir, 'train', 'training.scd.gold'), 'w'), indent=4)
        exs, labs = semshift2wic('rusemshift-tsvs/dev_1.tsv', 'dev.scd_1', mode=mode)
        json.dump(exs, open(os.path.join(data_dir, 'dev', 'dev.scd_1.data'), 'w'), indent=4)
        json.dump(labs, open(os.path.join(data_dir, 'dev', 'dev.scd_1.gold'), 'w'), indent=4)
        exs, labs = semshift2wic('rusemshift-tsvs/dev_2.tsv', 'dev.scd_2', mode=mode)
        json.dump(exs, open(os.path.join(data_dir, 'dev', 'dev.scd_2.data'), 'w'), indent=4)
        json.dump(labs, open(os.path.join(data_dir, 'dev', 'dev.scd_2.gold'), 'w'), indent=4)

    os.system('rm -rf summer-lsc')
    os.makedirs('rusemshift-ruwic-data', exist_ok=True)
    os.system('cp -r rusemshift-data/* rusemshift-ruwic-data/')
    for mode in ['mean', 'median']:
        files = glob('data/train/*ru-ru*')
        for file in files:
            os.system(f'cp {file} rusemshift-ruwic-data/{mode}/train/')
        files = glob('data/dev/*ru-ru*')
        for file in files:
            os.system(f'cp {file} rusemshift-ruwic-data/{mode}/dev/')
    

    os.system('mv data wic')
    os.makedirs('data')
    os.system('mv wic data/')

    os.system('mv rusemshift-ruwic-data data/')
    os.system('mv rusemshift-data data/')


def semshift2wic(semshift_raw, set_prefix, mode='median', split='dev'):
    if split == 'dev':
        # we want original mean scores for dev
        mode = 'mean'
    df = pd.read_csv(semshift_raw, sep='\t')
    data, data_labels = [], []

    def extract_spans(string):
        regex = re.compile(r'(?P<start><b><i>)(?P<word>\w+)(?P<nonalpha>\W*)(?P<end><\/i><\/b>)')
        spans = []
        while True:
            match = regex.search(string)
            if match is None:
                break
            start = len(string[:match.start('start')])
            end = match.end('word') - len('<b><i>')
            string = \
                string[:match.start('start')] + \
                string[match.start('word'):match.end('nonalpha')] + \
                string[match.end('end'):]
            spans.append((start, end))
        return spans, string

    ex_id = 0
    for row_id, row in enumerate(df.itertuples()):
        lemma = row.word
        pos = 'NOUN'
        sent1_spans, sent1 = extract_spans(row.sent1)
        sent2_spans, sent2 = extract_spans(row.sent2)
        for (s1, e1) in sent1_spans:
            for (s2, e2) in sent2_spans:
                idx = f'{set_prefix}.{ex_id}'
                ex_id += 1
                example = {
                    'id': idx,
                    'lemma': lemma,
                    'pos': pos,
                    'sentence1': sent1,
                    'sentence2': sent2,
                    'start1': s1,
                    'end1': e1,
                    'start2': s2,
                    'end2': e2,
                    'grp': row.group
                }
                if mode == 'median':
                    score = sorted([row.annotator1, row.annotator2, row.annotator3, row.annotator4, row.annotator5])[2]
                elif mode == 'mean':
                    score = row.mean

                if score >= 3.0:
                    tag = 'T'
                elif score <= 2.0:
                    tag = 'F'
                else:
                    continue

                label = {
                    'id': idx,
                    'tag': tag,
                    'row': row_id,
                    'score': score
                }
                data.append(example)
                data_labels.append(label)

    return data, data_labels


if __name__ == '__main__':
    random.seed(2021)
    fire.Fire(prepare_dataset)
