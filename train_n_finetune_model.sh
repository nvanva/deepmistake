DATA_TRAIN_DIR=$1 #data/wic/
DATA_FT_DIR=$2 #data/rusemshift-data/mean/
train_loss=$3 #crossentropy_loss
ft_loss=$4 #mse_loss
pool=$5 #mean
targ_emb=$6 #concat
batch_norm=$7 #1
hs=$8 #300
train_epochs=$9 #50
ft_epochs=${10} #50
grad_acc_steps=${11} #4
train_ckpt=${12} #nen-nen-weights or accuracy.dev.en-en.score

OUTPUT_DIR=xlmrlarge-train_loss-${train_loss}-ft_loss-${ft_loss}-pool-${pool}-targ_emb-${targ_emb}-hs-${hs}-bn-${batch_norm}-ckpt-${train_ckpt}
linhead=$([ "$hs" == 0 ] && echo "true" || echo "false")
echo output_dir = $OUTPUT_DIR
echo data_train = $DATA_TRAIN_DIR, data_ft = $DATA_FT_DIR
echo train_epochs = $train_epochs, ft_epochs = $ft_epochs, grad_acc_steps = $grad_acc_steps

python run_model.py --do_train --do_validation --data_dir $DATA_TRAIN_DIR --output_dir ${OUTPUT_DIR}/train/ --gradient_accumulation_steps $grad_acc_steps \
	--pool_type $pool --target_embeddings $targ_emb --head_batchnorm $batch_norm --loss $train_loss --linear_head ${linhead} --head_hidden_size $hs \
	--num_train_epochs $train_epochs
python run_model.py --train_scd --do_train --do_validation --data_dir $DATA_FT_DIR --output_dir ${OUTPUT_DIR}/finetune/ --gradient_accumulation_steps $grad_acc_steps \
	--pool_type $pool --target_embeddings $targ_emb --head_batchnorm $batch_norm --loss $ft_loss --linear_head ${linhead} --head_hidden_size $hs \
	--num_train_epochs $ft_epochs --ckpt_path ${OUTPUT_DIR}/train/${train_ckpt}/