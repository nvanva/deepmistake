ckpt_path=$1  # ABSOLUTE path to directory with checkpoint, args.json and pytorch_model.bin must be in {ckpt_path} directory!
echo ckpt_path=$ckpt_path
part=sampled_test_RuShiftEval
python run_model.py --do_eval --ckpt_path $ckpt_path --eval_input_dir data_dumped_full/$part/ \
  --eval_output_dir $ckpt_path/scores/$part/ --output_dir $ckpt_path
