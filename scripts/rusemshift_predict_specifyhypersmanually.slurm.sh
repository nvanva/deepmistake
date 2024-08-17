#!/bin/bash
#SBATCH --gres=gpu:1
ckpt_path=$(dirname $1)
for part in train dev; do 
python run_model.py --do_eval --ckpt_path $ckpt_path --eval_input_dir ./data/rusemshift-data/mean/${part} --eval_output_dir rusemshift_${ckpt_path}/${part} --output_dir preds-rusemshift ${@:2}
done


