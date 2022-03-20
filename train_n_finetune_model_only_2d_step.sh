#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
train_loss=$1 #crossentropy_loss
ft_loss=$2 #mse_loss
pool=$3 #mean
targ_emb=$4 #concat
batch_norm=$5 #1
hs=$6 #300
train_epochs=$7 #50
ft_epochs=$8 #50
grad_acc_steps=$9 #4
train_ckpt=${10} #nen-nen-weights or accuracy.dev.en-en.score
OUTPUT_DIR=${11} #xlmrlarge-train_loss-${train_loss}-ft_loss-${ft_loss}-pool-${pool}-targ_emb-${targ_emb}-hs-${hs}-bn-${batch_norm}-ckpt-${train_ckpt}
DATA_TRAIN_DIR=${12} #data/wic/
#ft_save_by_score=${13} #spearman.dev.scd_1.score+spearman.dev.scd_2.score+spearman.dev.scd_1.wordwise.score+spearman.dev.scd_2.wordwise.score
DATA_FT_DIR=${13} #data/rusemshift-data/

linhead=$([ "$hs" == 0 ] && echo "true" || echo "false")
train_scd=--train_scd
if [ $ft_loss = 'crossentropy_loss' ]; then
	train_scd=''
fi
echo output_dir = $OUTPUT_DIR
echo data_train = $DATA_TRAIN_DIR, data_ft = $DATA_FT_DIR
echo train_epochs = $train_epochs, ft_epochs = $ft_epochs, grad_acc_steps = $grad_acc_steps


if [ -n "$DATA_FT_DIR" ]; then
	python run_model.py $train_scd --do_train --do_validation --data_dir $DATA_FT_DIR --output_dir ${OUTPUT_DIR}/finetune/ --gradient_accumulation_steps $grad_acc_steps \
		--pool_type $pool --target_embeddings $targ_emb --head_batchnorm $batch_norm --loss $ft_loss --linear_head ${linhead} --head_hidden_size $hs \
		--num_train_epochs $ft_epochs --ckpt_path ${train_ckpt}/ 
else
	echo Without fine-tuning
fi
