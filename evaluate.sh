#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00

ckpt_path=nen-nen-weights
path_to_pairs=data_dumped_full/all_part_pairs
output_dir=nen-nen-weights

for num_part in 0 1 2 3 4 5 6 7 8 9; do
bash python run_model.py --do_eval --ckpt_path $ckpt_path --eval_input_dir ${path_to_pairs}/pairs_part_${num_part} --eval_output_dir dwug_predictions_${pairs_part}/ --output_dir $output_dir --loss crossentropy_loss --pool_type mean --symmetric true --train_scd --head_batchnorm 1 --linear_head true --head_hidden_size 0 --target_embeddings dist_l1ndotn
done
