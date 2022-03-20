#!/bin/bash
train_ckpt=$1
OUTPUT_DIR=$2
ft_data=$3
ft_save_by_score=$4
pool=mean
targ_emb=dist_l1ndotn
ft_loss=crossentropy_loss
linhead=true
python run_model.py --do_train --do_validation --data_dir $ft_data --output_dir $OUTPUT_DIR --gradient_accumulation_steps 16 \
	--pool_type $pool --target_embeddings $targ_emb --head_batchnorm 1 --loss $ft_loss --linear_head $linhead \
	--num_train_epochs 50  --symmetric true --save_by_score $ft_save_by_score \
	--model_name xlm-roberta-large --ckpt_path $train_ckpt
