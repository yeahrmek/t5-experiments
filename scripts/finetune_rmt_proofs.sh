task_name=proofs
data_dir="../lean_data/full_proofs_with_args/"  # or "../lean_data/source_code_rmt/"
wandb_project=rmt_proofs  # or lean
id=null
logger_resume=false
resume_training=false
pretrained_ckpt="./logs/rmt_proofs/uwbr70yj/checkpoints/last-v1.ckpt"
# pretrained_ckpt="./logs/rmt_lean/6rqkrhxp/checkpoints/last.ckpt"
# pretrained_ckpt="./logs/rmt_lean/7xxtf5sm/checkpoints/n_segments=4-epoch=00-step=156-loss=0.8870.ckpt"
# pretrained_ckpt="./logs/rmt_lean/6rqkrhxp/checkpoints/"

num_mem_tokens=10
# curriculum="[1,1,1,2,1,3,1,4,1,5]"
curriculum="[3,2,3,3,3,4,3,5]"

lr=1e-5


python run_finetuning_lean_pl.py \
  --task_name $task_name \
  --logger.save_dir ./logs \
  --logger.project $wandb_project \
  --logger.entity yeahrmek \
  --logger.id $id \
  --logger.resume $logger_resume \
  --resume_training $resume_training \
  --pretrained_ckpt $pretrained_ckpt \
  --data_dir $data_dir \
  --input_size 2048 \
  --tokenizer  ../mix2_tokenizer.ckpt \
  --backbone_cls transformers:GPTNeoForCausalLM \
  --backbone_cpt ../mix2_2zywvs69.ckpt \
  --rmt_cls modeling_rmt:RMTDecoderForCausalLM \
  --num_mem_tokens $num_mem_tokens \
  --max_n_segments 1 \
  --curriculum=$curriculum \
  --batch_size 1 \
  --optimizer.lr $lr \
  --lr_scheduler.warmup_epochs 1000 \
  --trainer.accumulate_grad_batches 32 \
  --trainer.limit_val_batches null \
  --trainer.val_check_interval 5000 \
  --trainer.precision bf16 \
  --trainer.num_sanity_val_steps 1 \
  --trainer.accelerator auto \
  --trainer.devices auto \
  --trainer.strategy deepspeed_stage_2