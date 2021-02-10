import fire
import random
import os
import json
from glob import glob


def prepare_dataset(dev_prop_to_train: int = 0.7, exclude_devs_from_split: str = 'dev.en-en', clean_all: bool = False):
	if clean_all:
		os.system('rm -rf data tmp')
		return
	os.system('git clone https://github.com/SapienzaNLP/mcl-wic.git')
	os.system('unzip mcl-wic/SemEval-2021_MCL-WiC_trial.zip -d tmp')
	os.system('unzip mcl-wic/SemEval-2021_MCL-WiC_all-datasets.zip -d tmp')
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
		part = file.split('/')[-1].split('.')[1]
		# os.makedirs(f'{test_dir}{part}', exist_ok=True)
		os.system(f'mv {file} {test_dir}/')

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


if __name__ == '__main__':
	random.seed(2021)
	fire.Fire(prepare_dataset)