grad_acc_steps=$1
train_epochs=50
ft_epochs=50
DATA_TRAIN_DIR=data_dumped_full/wic/
DATA_FT_DIR=data_dumped_full/rusemshift-data/
train_loss=crossentropy_loss
ft_loss=mse_loss
targ_emb=dist_l1ndotn
hs=0
pool=mean
batch_norm=1
train_ckpt=nen-nen-weights

# wic_train-en-en, wic full, wic ru-ru
for dataset in wic_train-en-en wic wic_ru-ru; do
DATA_TRAIN_DIR=data_dumped_full/${dataset}/
OUTPUT_DIR=xlmr-large..data_train-${dataset}..train_loss-${train_loss}..pool-${pool}..targ_emb-${targ_emb}..hs-${hs}..bn-${batch_norm}
bash train_n_finetune_model.sh $train_loss $ft_loss $pool $targ_emb $batch_norm $hs $train_epochs $ft_epochs $grad_acc_steps $train_ckpt $OUTPUT_DIR $DATA_TRAIN_DIR
done

# mcl-wic full -> [wic ru-ru, rusemshift, rusemshift + wic ru-ru] on CE loss
DATA_TRAIN_DIR=data_dumped_full/wic/
ft_loss=crossentropy_loss
for dataset in wic_ru-ru rusemshift-data rusemshift-ruwic-data; do
if [ $(echo $dataset | grep rusemshift) ]; then
	ft_save_by_score=spearman.dev.scd_1.score+spearman.dev.scd_2.score+spearman.dev.scd_1.wordwise.score+spearman.dev.scd_2.wordwise.score
else
	ft_save_by_score=accuracy.dev.nen-nen.score
fi
DATA_FT_DIR=data_dumped_full/${dataset}/
OUTPUT_DIR=xlmr-large..data_train-wic..train_loss-${train_loss}..data_ft-${dataset}..ft_loss-${ft_loss}..pool-${pool}..targ_emb-${targ_emb}..hs-${hs}..bn-${batch_norm}..ckpt-${train_ckpt}
bash train_n_finetune_model.sh $train_loss $ft_loss $pool $targ_emb $batch_norm $hs $train_epochs $ft_epochs $grad_acc_steps $train_ckpt $OUTPUT_DIR $DATA_TRAIN_DIR $ft_save_by_score $DATA_FT_DIR
done

ft_save_by_score=spearman.dev.scd_1.score+spearman.dev.scd_2.score+spearman.dev.scd_1.wordwise.score+spearman.dev.scd_2.wordwise.score
# mcl-wic full -> RuSemShift (mse loss)
ft_loss=mse_loss
dataset=rusemshift-data
DATA_FT_DIR=data_dumped_full/${dataset}/
OUTPUT_DIR=xlmr-large..data_train-wic..train_loss-${train_loss}..data_ft-${dataset}..ft_loss-${ft_loss}..pool-${pool}..targ_emb-${targ_emb}..hs-${hs}..bn-${batch_norm}..ckpt-${train_ckpt}
bash train_n_finetune_model.sh $train_loss $ft_loss $pool $targ_emb $batch_norm $hs $train_epochs $ft_epochs $grad_acc_steps $train_ckpt $OUTPUT_DIR $DATA_TRAIN_DIR $ft_save_by_score $DATA_FT_DIR

# mcl-wic full -> RuSemShift + wic ru-ru (mse+ loss)
ft_loss=mseplus_loss
dataset=rusemshift-ruwic-data
DATA_FT_DIR=data_dumped_full/${dataset}/
OUTPUT_DIR=xlmr-large..data_train-wic..train_loss-${train_loss}..data_ft-${dataset}..ft_loss-${ft_loss}..pool-${pool}..targ_emb-${targ_emb}..hs-${hs}..bn-${batch_norm}..ckpt-${train_ckpt}
bash train_n_finetune_model.sh $train_loss $ft_loss $pool $targ_emb $batch_norm $hs $train_epochs $ft_epochs $grad_acc_steps $train_ckpt $OUTPUT_DIR $DATA_TRAIN_DIR $ft_save_by_score $DATA_FT_DIR

# RuSemShift (MSE loss)
train_loss=mse_loss
dataset=rusemshift-data
DATA_TRAIN_DIR=data_dumped_full/${dataset}/
OUTPUT_DIR=xlmr-large..data_train-${dataset}..train_loss-${train_loss}..pool-${pool}..targ_emb-${targ_emb}..hs-${hs}..bn-${batch_norm}
