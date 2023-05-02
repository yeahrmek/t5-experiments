id=null
logger_resume=false
resume_training=false
# pretrained_ckpt="./logs/rmt_lean/bfv00lo5/checkpoints/n_segments=3-epoch=00-step=468-loss=0.9029.ckpt"
pretrained_ckpt=null

num_mem_tokens=10
curriculum="[2,1,1,2,1,3,1,4,1,5]"
# curriculum="[1,2,1,3,1,4,1,5]"
# curriculum="[1,4,1,5]"

lr=1e-5


python run_finetuning_lean_pl.py \
  --task_name lean \
  --logger.save_dir ./logs \
  --logger.project rmt_lean \
  --logger.entity yeahrmek \
  --logger.id $id \
  --logger.resume $logger_resume \
  --resume_training $resume_training \
  --pretrained_ckpt $pretrained_ckpt \
  --data_dir ../lean_data/source_code_rmt/ \
  --input_size 2048 \
  --tokenizer  ../mix2_tokenizer.ckpt \
  --backbone_cls transformers:GPTNeoForCausalLM \
  --backbone_cpt ../mix2_2zywvs69.ckpt \
  --rmt_cls modeling_rmt:RMTDecoderForCausalLM \
  --num_mem_tokens $num_mem_tokens \
  --max_n_segments 1 \
  --curriculum=$curriculum \
  --batch_size 2 \
  --optimizer.lr $lr \
  --lr_scheduler.warmup_epochs 1000 \
  --trainer.accumulate_grad_batches 16 \
  --trainer.limit_val_batches null \
  --trainer.val_check_interval 5000 \
  --trainer.precision bf16 \
  --trainer.num_sanity_val_steps 1 \
  --trainer.accelerator auto \
  --trainer.devices auto \
  --trainer.strategy deepspeed_stage_2
