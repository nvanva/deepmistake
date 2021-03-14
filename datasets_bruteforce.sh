train_epochs=$1 #50
ft_epochs=$2 #50
DATA_TRAIN_DIR=data_dumped_full/wic/
DATA_FT_DIR=data_dumped_full/rusemshift-data/
grad_acc_steps=4
train_loss=crossentropy_loss
ft_loss=mse_loss
targ_emb=concat
hs=-1
pool=mean
batch_norm=1
train_ckpt=nen-nen-weights

# wic_train-en-en, wic full, wic ru-ru
for dataset in wic_train-en-en wic wic_ru-ru; do
DATA_TRAIN_DIR=data_dumped_full/${dataset}/
OUTPUT_DIR=xlmrlarge-data_train-${dataset}-train_loss-${train_loss}-ft_loss-${ft_loss}-pool-${pool}-targ_emb-${targ_emb}-hs-${hs}-bn-${batch_norm}-ckpt-${train_ckpt}	
bash train_n_finetune_model.sh $train_loss $ft_loss $pool $targ_emb $batch_norm $hs $train_epochs $ft_epochs $grad_acc_steps $train_ckpt $OUTPUT_DIR $DATA_TRAIN_DIR
done

# mcl-wic full -> [wic ru-ru, rusemshift, rusemshift + wic ru-ru] on CE loss
DATA_TRAIN_DIR=data_dumped_full/wic/
ft_loss=crossentropy_loss
for dataset in wic_ru-ru rusemshift-data rusemshift-ruwic-data; do
DATA_FT_DIR=data_dumped_full/${dataset}/
OUTPUT_DIR=xlmrlarge-data_train-wic-data_ft-${dataset}-train_loss-${train_loss}-ft_loss-${ft_loss}-pool-${pool}-targ_emb-${targ_emb}-hs-${hs}-bn-${batch_norm}-ckpt-${train_ckpt}	
bash train_n_finetune_model.sh $train_loss $ft_loss $pool $targ_emb $batch_norm $hs $train_epochs $ft_epochs $grad_acc_steps $train_ckpt $OUTPUT_DIR $DATA_TRAIN_DIR $DATA_FT_DIR
done

# mcl-wic full -> RuSemShift (mse loss)
ft_loss=mse_loss
dataset=rusemshift-data
DATA_FT_DIR=data_dumped_full/${dataset}/
OUTPUT_DIR=xlmrlarge-data_train-wic-data_ft-${dataset}-train_loss-${train_loss}-ft_loss-${ft_loss}-pool-${pool}-targ_emb-${targ_emb}-hs-${hs}-bn-${batch_norm}-ckpt-${train_ckpt}	
bash train_n_finetune_model.sh $train_loss $ft_loss $pool $targ_emb $batch_norm $hs $train_epochs $ft_epochs $grad_acc_steps $train_ckpt $OUTPUT_DIR $DATA_TRAIN_DIR $DATA_FT_DIR

# mcl-wic full -> RuSemShift + wic ru-ru (mse+ loss)
ft_loss=mseplus_loss
dataset=rusemshift-ruwic-data
DATA_FT_DIR=data_dumped_full/${dataset}/
bash train_n_finetune_model.sh $train_loss $ft_loss $pool $targ_emb $batch_norm $hs $train_epochs $ft_epochs $grad_acc_steps $train_ckpt $OUTPUT_DIR $DATA_TRAIN_DIR $DATA_FT_DIR
