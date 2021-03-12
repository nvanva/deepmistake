for lr in 1e-5 2e-5; do
for loss in mse_loss ; do
for pool in mean; do
for symmetric in true; do
for lr_s in linear_warmup ; do 
for tembs in comb_dmn ; do
for bn in 1 ; do
#for ckpt_path in trained_xlmrlarge-dmn/lr-2e-5-symmetric-true-lrs-linear_warmup-pool-mean-tembs-comb_dmn-bn-1/21-02-21T00-59-28-467365449/nen-nen-weights/ trained_xlmrlarge-dmn/lr-2e-5-symmetric-true-lrs-linear_warmup-pool-mean-tembs-comb_dmn-bn-1/21-02-21T00-59-28-467365449/ '' ; do
ckpt_path=NONE
OUT_DIR=trained_rusemshift-xlmrlarge-dmn/lr-$lr-symmetric-$symmetric-lrs-${lr_s}-pool-$pool-tembs-$tembs-bn-$bn/`date +"%y-%m-%dT%H-%M-%S-%N"`/$ckpt_path
sbatch run_model_slurm.sh --do_train --do_validation --lr_scheduler $lr_s \
                    --data_dir data/rusemshift-data/mean --output_dir $OUT_DIR \
                    --learning_rate $lr \
                    --loss $loss --train_scd  --save_by_score spearman.dev.scd_2.score \
                    --pool_type $pool --target_embeddings $tembs --head_batchnorm $bn \
                    --symmetric $symmetric \
                    --num_train_epochs 50 \
                    --warmup_proportion 0.1 \
                    --start_save_threshold 0.3 \
                    --gradient_accumulation_steps 4;
#                echo python run_model.py --do_eval \
#                    --data_dir data/wic/ --output_dir $OUT_DIR \
#                    --learning_rate $lr \
#                    --loss $loss \
#                    --pool_type $pool \
#                    --symmetric $symmetric;
#done
done
done
done
done
done
done
done

