ckpt_path=$1  # path to directory with checkpoint, args.json and pytorch_model.bin must be in {ckpt_path} directory!
for part in test_RuShiftEval rusemshift-data/dev rusemshift-data/train; do
python run_model.py --do_eval --ckpt_path $ckpt_path --eval_input_dir data_dumped_full/$part/ \
  --eval_output_dir $ckpt_path/scores/$part/ --output_dir $ckpt_path
done
