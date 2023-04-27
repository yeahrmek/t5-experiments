python run_finetuning_lm_rmt_lean.py \
  --task_name lean \
  --wandb_project rmt_lean \
  --log_dir ./logs \
  --data_dir ../lean_data/source_code_rmt/ \
  --input_size 2048 \
  --tokenizer  ../mix2_tokenizer.ckpt \
  --backbone_cls transformers:GPTNeoForCausalLM \
  --backbone_cpt ../mix2_2zywvs69.ckpt \
  --rmt_cls modeling_rmt:RMTDecoderForCausalLM \
  --num_mem_tokens 10 \
  --max_n_segments 1 \
  --curriculum 29000 1 28000 2 26000 3 24000 4 22000 5 \
  --lr 1e-5 \
  --lr_scheduler linear \
  --num_warmup_steps 1000 \
  --gradient_accumulation_steps 16 \
  --batch_size 2 \
  --valid_interval 5000 \
  --log_interval 1 \
  --save_best

#  --save_interval 50000 \