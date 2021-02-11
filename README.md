# mcl-wic
To install dependencies:
```bash
pip install -r requirements.txt
```
To prepare dataset:

```bash
python prepare_dataset.py
```

To run ot train the model, adjust local_config.json and run command:
```bash
python run_model.py --local_config_path local_config.json --output_dir MODEL_OUT_DIR
```
or
```bash
python run_qa_model.py --local_config_path local_config.json --output_dir MODEL_OUT_DIR
```
