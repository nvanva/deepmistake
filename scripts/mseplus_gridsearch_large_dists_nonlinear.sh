for lr in 1e-5 2e-5; do
for loss in mseplus_loss ; do
for pool in mean; do
for symmetric in true; do
for lr_s in linear_warmup ; do 
for tembs in dist_l1 dist_l1ndotn ; do
for hs in 0 100 300 ; do
for bn in 1 ; do
for agg in mean ; do
for ckpt_path in trained_xlmrlarge-dmn/lr-2e-5-symmetric-true-lrs-linear_warmup-pool-mean-tembs-${tembs}-bn-1-linhead-True/*/nen-nen-weights/ ; do
#ckpt_path=NONE
linhead=$([ "$hs" == 0 ] && echo "true" || echo "false")
OUT_DIR=trained_rusemshift-xlmrlarge-dmn/lr-$lr-symmetric-$symmetric-lrs-${lr_s}-pool-$pool-tembs-$tembs-bn-$bn-hs${hs}-linhead${linhead}-mseplusV6${agg}/`date +"%y-%m-%dT%H-%M-%S-%N"`/$ckpt_path
sbatch run_model_slurm.sh --do_train --do_validation --lr_scheduler $lr_s \
                    --data_dir data_dumped_full/rusemshift-ruwic-data/${agg} --output_dir $OUT_DIR --ckpt_path ${ckpt_path} \
                    --learning_rate $lr \
                    --loss $loss --train_scd  --save_by_score spearman.dev.scd_1.score+spearman.dev.scd_2.score+spearman.dev.scd_1.wordwise.score+spearman.dev.scd_2.wordwise.score \
                    --pool_type $pool --target_embeddings $tembs --head_batchnorm $bn --linear_head ${linhead} --head_hidden_size $hs \
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
done

done
done
done
done
done
done
done
done
done

