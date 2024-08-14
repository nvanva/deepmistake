train_epochs=$1 #50
ft_epochs=$2 #50
DATA_TRAIN_DIR=data_dumped_full/wic/
DATA_FT_DIR=data_dumped_full/rusemshift-data/
grad_acc_steps=4
train_loss=crossentropy_loss
ft_loss=mse_loss
pool=mean
batch_norm=1
train_ckpt=nen-nen-weights
ft_save_by_score=spearman.dev.scd_1.score+spearman.dev.scd_2.score+spearman.dev.scd_1.wordwise.score+spearman.dev.scd_2.wordwise.score

for targ_emb in dist_l1ndotn dist_l1; do
for hs in 0 300; do
OUTPUT_DIR=xlmrlarge-train_loss-${train_loss}-ft_loss-${ft_loss}-pool-${pool}-targ_emb-${targ_emb}-hs-${hs}-bn-${batch_norm}
bash train_n_finetune_model.sh $train_loss $ft_loss $pool $targ_emb $batch_norm $hs $train_epochs $ft_epochs $grad_acc_steps $train_ckpt $OUTPUT_DIR $DATA_TRAIN_DIR \
	$ft_save_by_score $DATA_FT_DIR
done
done

for targ_emb in concat comb_dmn; do
for hs in -1; do
OUTPUT_DIR=xlmrlarge-train_loss-${train_loss}-ft_loss-${ft_loss}-pool-${pool}-targ_emb-${targ_emb}-hs-${hs}-bn-${batch_norm}
bash train_n_finetune_model.sh $train_loss $ft_loss $pool $targ_emb $batch_norm $hs $train_epochs $ft_epochs $grad_acc_steps $train_ckpt $OUTPUT_DIR $DATA_TRAIN_DIR \
	$ft_save_by_score $DATA_FT_DIR
done
done
