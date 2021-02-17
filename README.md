# mcl-wic

To install dependencies:
```bash
pip install -r requirements.txt
```
To prepare dataset:

```bash
python prepare_dataset.py
```

To train the model:
```bash
python run_model.py --do_train --do_validation --data_dir data/wic/ --output_dir MODEL_OUT_DIR
```


To run grid searh:
```bash
for lr in 1e-5 2e-5 5e-6;
do
    for loss in mse_loss crossentropy_loss;
    do
        for pool in max first;
        do
            for symmetric in true false;
            do
                for lr_s in linear_warmup constant_warmup;
                do
                    OUT_DIR=trained_models/lr-$lr-loss-$loss-pool-$pool-symmetric-$symmetric-lrs-${lr_s}/
                    echo python run_model.py --do_train --do_validation \
                        --data_dir data/wic/ --output_dir $OUT_DIR \
                        --learning_rate $lr \
                        --loss $loss \
                        --pool_type $pool \
                        --symmetric $symmetric \
                        --lr_scheduler $lr_s \
                        --num_train_epochs 50 \
                        --start_save_threshold 0.7 \
                        --gradient_accumulation_steps 16;
                    echo python run_model.py --do_eval \
                        --data_dir data/wic/ --output_dir $OUT_DIR \
                        --learning_rate $lr \
                        --loss $loss \
                        --pool_type $pool \
                        --symmetric $symmetric;
                done
            done
        done
    done
done
```
