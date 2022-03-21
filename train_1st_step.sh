#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
train_loss=crossentropy_loss
pool=mean
targ_emb=dist_l1ndotn
batch_norm=1
hs=0
train_epochs=50
grad_acc_steps=4
DATA_TRAIN_DIR=data_dumped_full/wic+dwug+xl-wsd/
OUTPUT_DIR=l1ndotn_es_lscd/xlmrlarge-train_loss-${train_loss}-pool-${pool}-targ_emb-${targ_emb}-hs-${hs}-bn-${batch_norm}-train_data-wic+dwug+xlwsd
ft_save_by_score=spearman.dev.scd_1.score+spearman.dev.scd_1.wordwise.score

linhead=$([ "$hs" == 0 ] && echo "true" || echo "false")
train_scd=--train_scd
if [ $ft_loss = 'crossentropy_loss' ]; then
	train_scd=''
fi
echo output_dir = $OUTPUT_DIR
echo data_train = $DATA_TRAIN_DIR
echo train_epochs = $train_epochs, grad_acc_steps = $grad_acc_steps

python run_model.py --do_train --do_validation --data_dir $DATA_TRAIN_DIR --output_dir ${OUTPUT_DIR}/train/ --gradient_accumulation_steps $grad_acc_steps \
	--pool_type $pool --target_embeddings $targ_emb --head_batchnorm $batch_norm --loss $train_loss --linear_head ${linhead} --head_hidden_size $hs \
	--num_train_epochs $train_epochs

