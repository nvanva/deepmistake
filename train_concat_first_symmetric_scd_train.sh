train_epochs=50
ft_epochs=30
DATA_TRAIN_DIR=data_dumped_full/wic/
DATA_FT_DIR=data_dumped_full/rusemshift-data/mean/
grad_acc_steps=4
train_loss=crossentropy_loss
ft_loss=mse_loss
pool=first
batch_norm=0
targ_emb=concat
hs=-1
train_ckpt=accuracy.dev.en-en.score
bash train_n_finetune_model.sh $train_loss $ft_loss $pool $targ_emb $batch_norm $hs $train_epochs $ft_epochs $grad_acc_steps $train_ckpt $DATA_TRAIN_DIR $DATA_FT_DIR