task_name=proofs
data_dir="../lean_data/full_proofs_with_args/"
lemmas_path="../lean_data/all_lemmas_statements.json"
wandb_project=rmt_proofs_only  # or lean
id=null
logger_resume=false
resume_training=false
#pretrained_ckpt="./logs/rmt_proofs_only/j3ayk2sg/checkpoints/last.ckpt"
pretrained_ckpt=null

proof_loss_only=false
short_proofs_only=true
every_segment_def=true
exclude_relevant_lemmas=false
use_random_lemmas_names=true

use_recur_mem=true

def_lemmas_loss_weight=0.5
proof_loss_weight=0.5

num_mem_tokens=10
# curriculum="[3,2,2,3,2,4,2,5,1,6,1,7]"
curriculum="[7,2,2,3,2,4,2,5,1,6,1,7]"
#curriculum="[7,2,2,3,2,4,2,5]"
#curriculum=[6,7]
#curriculum=[7,2,2,3,2,4,2,5,2,2,1,3,1,4,1,5]
model_type="rmt"

lr=1e-5

input_size=512
batch_size=2
accumulate_grad_batches=16

export CUDA_VISIBLE_DEVICES=0

python run_finetuning_lean_pl.py \
  --task_name $task_name \
  --model_type $model_type \
  --logger.save_dir ./logs \
  --logger.project $wandb_project \
  --logger.entity versham \
  --logger.id $id \
  --logger.resume $logger_resume \
  --resume_training $resume_training \
  --pretrained_ckpt $pretrained_ckpt \
  --data_dir $data_dir \
  --lemmas_path $lemmas_path \
  --short_proofs_only $short_proofs_only \
  --every_segment_def $every_segment_def \
  --exclude_relevant_lemmas $exclude_relevant_lemmas \
  --use_random_lemmas_names $use_random_lemmas_names \
  --proof_loss_only $proof_loss_only \
  --def_lemmas_loss_weight $def_lemmas_loss_weight \
  --proof_loss_weight $proof_loss_weight \
  --use_recur_mem $use_recur_mem \
  --input_size $input_size \
  --tokenizer  ../mix2_tokenizer.ckpt \
  --backbone_cls transformers:GPTNeoForCausalLM \
  --backbone_cpt ../mix2_2zywvs69.ckpt \
  --rmt_cls modeling_rmt:RMTDecoderForCausalLM \
  --num_mem_tokens $num_mem_tokens \
  --max_n_segments 2 \
  --curriculum=$curriculum \
  --batch_size $batch_size \
  --data_n_workers 4 \
  --optimizer.lr $lr \
  --lr_scheduler.warmup_epochs 125 \
  --trainer.accumulate_grad_batches $accumulate_grad_batches \
  --trainer.limit_val_batches null \
  --trainer.val_check_interval 4950 \
  --trainer.precision bf16-mixed \
  --trainer.num_sanity_val_steps 1 \
  --trainer.accelerator auto \
  --trainer.devices auto
#  --trainer.strategy deepspeed_stage_2_offload
