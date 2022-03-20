grad_acc_steps=4
train_epochs=50
ft_epochs=50
DATA_TRAIN_DIR=data_dumped_full/wic/
DATA_FT_DIR=data_dumped_full/rusemshift-data/
train_loss=crossentropy_loss
targ_emb=dist_l1ndotn
ft_loss=crossentropy_loss
hs=0
pool=mean
batch_norm=1
train_ckpt=nen-nen_weights
ft_save_by_score = spearman.dev.dev.score

# mcl-wic full -> [wic ru-ru, rusemshift, rusemshift + wic ru-ru] on CE loss
DATA_TRAIN_DIR=data_dumped_full/wic/
for dataset in dwug_es_all_tp_bin1 dwug_es_all_tp_bin2 dwug_es_only_COMPARE_bin1 dwug_es_only_COMPARE_bin2 dwug_es+xl-wsd; do
DATA_FT_DIR=data_dumped_full/${dataset}/
OUTPUT_DIR=l1ndotn_es_lscd/xlmr-large..data_train-wic..train_loss-crossentropy_loss..data_ft-${dataset}..ft_loss-${ft_loss}..pool-${pool}..targ_emb-${targ_emb}..hs-${hs}..bn-${batch_norm}..ckpt-${train_ckpt}
bash train_n_finetune_model_only_2d_step.sh $train_loss $ft_loss $pool $targ_emb $batch_norm $hs $train_epochs $ft_epochs $grad_acc_steps $train_ckpt $OUTPUT_DIR $DATA_TRAIN_DIR $ft_save_by_score $DATA_FT_DIR
done

