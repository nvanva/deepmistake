# DeepMistake-WiC
This repository contains **DeepMistake-WiC**, an improved version of the Word-in-Context model used by the DeepMistake team.

**DeepMistake** is the name of the team that participated in two Lexical Semantic Change Detections (LSCD) shared tasks: [DeepMistake at RuShiftEval-2021](https://www.dialog-21.ru/media/5491/arefyevnplusetal133.pdf) for Russian and [DeepMistake at LSCDiscovery-2022](https://aclanthology.org/2022.lchange-1.18/) for Spanish. The main components of the proposed solutions are: 
- sampling pairs of word usages,
- training and applying a Word-in-Context (WiC) model,
- aggregation of the predicted WiC scores.

If your goal is reproducing the results from these shared tasks, look at [DeepMistake at RuShiftEval-2021](https://github.com/Daniil153/DeepMistake) and [DeepMistake and LSCDiscovery-2022](https://github.com/Daniil153/DM-in-Spanish-LSCDiscovery).

The (originally unnamed) WiC model employed by the **DeepMistake** team was first developed for the Multilingual and Cross-Lingual Word-in-Context (MCL-WiC) shared task at SemEval-2021, then the architecture and training schema were improved and adapted for the LSCD tasks. Main results are published in:
- [Adis Davletov, Nikolay Arefyev, Denis Gordeev, Alexey Rey. LIORI at SemEval-2021 Task 2: Span Prediction and Binary Classification approaches to Word-in-Context Disambiguation, 2021](https://aclanthology.org/2021.semeval-1.103/)
- [Arefyev Nikolay, Maksim Fedoseev, Vitaly Protasov, Daniil Homskiy, Adis Davletov, Alexander Panchenko. DeepMistake: Which Senses are Hard to Distinguish for a Word­in­Context Model, 2021](https://www.dialog-21.ru/media/5491/arefyevnplusetal133.pdf)

This repository further develops the original WiC model. The main changes are:
- support of long usages: the original version just skipped pairs of usages longer than XLM-R can encode, DeepMistake-WiC intelligently trims usages to fit the encoder limitations while keeping both left and right context of substantial length for each usage;
- a Python package: DeepMistake-WiC can be installed and used as a Python package as well as by running the original inference scripts.

# Installation
First, clone the repository:
```bash
git clone https://github.com/Daniil153/DeepMistake
cd DeepMistake
git clone https://github.com/nvanva/deepmistake
cd deepmistake
```

To install dependencies:
```bash
pip install -r requirements.txt
```

To install the package:
```bash
pip install -e .
```

To prepare dataset:

```bash
python prepare_dataset.py
```

To use the model:
```python
from deepmistake.deepmistake import DeepMistake

dm_model = DeepMistake()

test_dir = "path/to/test/dataset"
output_dir = "path/to/output/directory"
eval_output_dir = "path/to/eval/output/directory" # The directory where features and labels will be saved will be output_dir/eval_output_dir

predictions = dm_model.predict_dataset(test_dir, output_dir, eval_output_dir)
```

To predict the test set:
```bash
bash eval_best_post-eval_model_dm.sh
```

# Training
To train the model:
```bash
python run_model.py --do_train --do_validation --data_dir DATA_DIR --output_dir MODEL_OUT_DIR
```
Examples of running model tranining: https://github.com/ameta13/mcl-wic
