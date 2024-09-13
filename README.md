# DeepMistakeWiC

First, clone the repository:
```bash
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
First, clone the LSCD Task:

```bash
cd .. # (if you are in the deepmistake repo)
git clone https://github.com/Daniil153/DeepMistake
mv ./deepmistak ./DeepMistake/deepmistake
cd DeepMistake
```

Then, run the following command.

```bash
bash eval_best_post-eval_model_dm.sh
```

To train the model:
```bash
python run_model.py --do_train --do_validation --data_dir DATA_DIR --output_dir MODEL_OUT_DIR
```
