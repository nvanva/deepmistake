train_epochs=$1 #50
ft_epochs=$2 #50
DATA_TRAIN_DIR=data_dumped_full/wic/
DATA_FT_DIR=data_dumped_full/rusemshift-data/mean/
grad_acc_steps=4
train_loss=crossentropy_loss
ft_loss=mse_loss
targ_emb=concat
hs=-1
pool=mean
batch_norm=1
train_ckpt=nen-nen-weights

# wic_train-en-en, wic full, wic ru-ru
for DATA_TRAIN_DIR in data_dumped_full/wic_train-en-en/ data_dumped_full/wic/ data_dumped_full/wic_ru-ru/; do
bash train_n_finetune_model.sh $train_loss $ft_loss $pool $targ_emb $batch_norm $hs $train_epochs $ft_epochs $grad_acc_steps $train_ckpt $DATA_TRAIN_DIR
done

# mcl-wic full -> [wic ru-ru, rusemshift, rusemshift + wic ru-ru] on CE loss
DATA_TRAIN_DIR=data_dumped_full/wic/
ft_loss=crossentropy_loss
for DATA_FT_DIR in data_dumped_full/wic_ru-ru/ data_dumped_full/rusemshift-data/mean/ data_dumped_full/rusemshift-ruwic-data/mean/; do
bash train_n_finetune_model.sh $train_loss $ft_loss $pool $targ_emb $batch_norm $hs $train_epochs $ft_epochs $grad_acc_steps $train_ckpt $DATA_TRAIN_DIR $DATA_FT_DIR
done

# mcl-wic full -> RuSemShift (mse loss)
ft_loss=mse_loss
DATA_FT_DIR=data_dumped_full/rusemshift-data/mean/
bash train_n_finetune_model.sh $train_loss $ft_loss $pool $targ_emb $batch_norm $hs $train_epochs $ft_epochs $grad_acc_steps $train_ckpt $DATA_TRAIN_DIR $DATA_FT_DIR

# mcl-wic full -> RuSemShift + wic ru-ru (mse+ loss)
ft_loss=mseplus_loss
DATA_FT_DIR=data_dumped_full/rusemshift-ruwic-data/mean/
bash train_n_finetune_model.sh $train_loss $ft_loss $pool $targ_emb $batch_norm $hs $train_epochs $ft_epochs $grad_acc_steps $train_ckpt $DATA_TRAIN_DIR $DATA_FT_DIR
