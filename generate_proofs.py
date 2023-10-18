import importlib
import logging
import os
from pathlib import Path
from pprint import pprint
from typing import List, Optional, Tuple

import torch
from jsonargparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoConfig, AutoTokenizer  # noqa: E402

from lean_dataset import (
    RMTDocsAllAtOnceDataLoader,
    RMTDocsDataLoader,
    RMTDocsDataset,
    RMTProofsDataset,
)
from modeling_rmt.lightning import RMTModelPL

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")


def get_cls_by_name(name: str) -> type:
    """Get class by its name and module path.

    Args:
        name (str): e.g., transfomers:T5ForConditionalGeneration, modeling_t5:my_class

    Returns:
        type: found class for `name`
    """
    module_name, cls_name = name.split(":")
    return getattr(importlib.import_module(module_name), cls_name)


def setup_parser():
    parser = ArgumentParser()
    parser.add_argument("--task_name", type=str, help="Task name: 'lm' or 'proofs'")
    parser.add_argument(
        "--model_type",
        type=str,
        default="rmt",
        help="rmt or base (no reccurency) model",
    )
    parser.add_argument("--proof_loss_only", type=bool, default=False)
    parser.add_argument("--short_proofs_only", type=bool, default=False)
    parser.add_argument("--every_segment_def", type=bool, default=True, 
                        help="if true definition is included into begining of each segment")
    parser.add_argument("--exclude_relevant_lemmas", type=bool, default=False)
    parser.add_argument("--use_random_lemmas_names", type=bool, default=False)
    parser.add_argument("--data_dir", type=str, help="Path to the data directory")
    parser.add_argument("--lemmas_path", type=str, help="Path to the JSON containing all lemmas statements")
    parser.add_class_arguments(WandbLogger, "logger")

    # For newer version of pytorch_lighning we add several parameters of logger explicitly
    try:
        parser.add_argument("--logger.resume", type=bool, default=False)
        parser.add_argument("--logger.entity", type=Optional[str], default=None)
    except:
        pass

    parser.add_argument("--resume_training", type=bool, default=False)
    parser.add_argument(
        "--validate_only",
        action="store_true",
        default=False,
        help="Skip training and run only validation. (default: False)",
    )
    parser.add_argument(
        "--working_dir",
        type=str,
        default=".",
        help="working dir, should be a dir with t5-experiments repo (default: .)",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--show_valid_examples",
        type=int,
        default=0,
        help="how many valid examples to show during training (default: 0)",
    )
    parser.add_argument(
        "--data_n_workers",
        type=int,
        default=1,
        help="number of dataloader workers (default: 2)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size",
    )

    parser.add_argument(
        "--input_prefix",
        type=str,
        default="",
        help='add task prefix to an input string (default: "")',
    )

    # model args
    parser.add_argument(
        "--rmt_cls",
        type=str,
        default="transformers:BertForPreTraining",
        help="RMT model class name to use (default: transformers:BertForPreTraining)",
    )
    parser.add_argument(
        "--pretrained_ckpt",
        type=Optional[str],
        default=None,
        help="pretrained model checkpoint path",
    )
    parser.add_argument(
        "--backbone_cls",
        type=str,
        default=None,
        help="backbone class name to use for RMT",
    )
    parser.add_argument(
        "--backbone_cpt",
        type=str,
        default=None,
        help="pretrained model checkpoint path",
    )

    # Aydar # RMT args
    parser.add_argument(
        "--input_size",
        type=int,
        default=None,
        help="maximal input size of the backbone model",
    )
    parser.add_argument(
        "--num_mem_tokens", type=int, default=None, help="number of memory tokens."
    )
    parser.add_argument(
        "--max_n_segments", type=int, default=1, help="maximal segment number"
    )
    parser.add_argument(
        "--curriculum",
        type=List[int],
        help="Scheduler for number of segments to train on. "
        "Input should be in the following format: <n_epochs> <n_segments> <n_epochs> <n_segments> ..."
        "Example: `--curriculum 1 1 1 2 2 5`. "
        "In this example we will first train for 1 epoch on 1 segment "
        "then train for 1 epoch on 2 segments, then for 2 epochs on 5 segments",
    )
    parser.add_argument(
        "--sum_loss",
        action="store_true",
        default=False,
        help="with this flag task loss from all segments is summed",
    )
    parser.add_argument(
        "--def_lemmas_loss_weight",
        type=float,
        default=0.0,
        help="weight of decl_def and lemmas loss in total loss"
    )
    parser.add_argument(
        "--proof_loss_weight",
        type=float,
        default=1.0,
        help="weight of proof loss in total loss"
    )
    parser.add_argument(
        "--use_recur_mem",
        type=bool,
        default=True,
        help="Use recurrent memory or not. If false, the model uses memory but only working on current segment."
    )
    parser.add_argument(
        "--bptt_depth",
        type=int,
        default=-1,
        help="max number of previous segments in gradient computation.",
    )
    parser.add_argument(
        "--segment_ordering",
        type=str,
        help="segment order",
        default="regular",
        choices=[
            "regular",
            "reversed",
            "bidirectional",
            "repeat_first",
            "last_memory_only",
        ],
    )
    parser.add_argument(
        "--memory_forward_func",
        type=str,
        help="path to memory forward funсtion script",
        default=None,
    )
    parser.add_argument(
        "--memory_layers",
        type=str,
        help='memory-augmented layer inds or "all" for all layers',
        default=None,
    )
    parser.add_argument(
        "--share_memory_layers",
        action="store_true",
        help="share weights of memory layers",
        default=False,
    )
    parser.add_argument(
        "--reconstruction_loss_coef",
        type=float,
        default=None,
        help="reconstuction loss ratio in total loss",
    )
    parser.add_argument(
        "--retain_graph",
        action="store_true",
        help="Retain computation graph during backward pass",
        default=False,
    )
    parser.add_argument(
        "--use_truncated_backward",
        action="store_true",
        default=False,
        help="whether to use RMT truncated bptt method in backward",
    )
    parser.add_argument(
        "--k1",
        type=int,
        default=-1,
        help="(not implemented) If not -1, gradient update is done each k1 segments",
    )
    parser.add_argument(
        "--k2", type=int, default=-1, help="number of last segments used by backward"
    )
    parser.add_argument(
        "--freeze_model_weights",
        action="store_true",
        default=False,
        help="Stop training all model weights except memory layers",
    )

    # tokenizer
    # todo: add wordpiece tokenizers support?
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="path or name of pre-trained HF Tokenizer",
    )

    # optimizer args
    parser.add_argument(
        "--optimizer.weight_decay",
        type=float,
        default=0.0,
        help="optimizer weight decay (default: 0.0)",
    )
    parser.add_argument(
        "--optimizer.lr",
        type=float,
        default=1e-5,
        help="learning rate",
    )
    parser.add_argument(
        "--optimizer.betas",
        type=Tuple[float, float],
        default=(0.9, 0.99),
        help="betas",
    )
    parser.add_argument(
        "--optimizer.eps",
        type=float,
        default=1e-8,
    )
    parser.add_argument(
        "--optimizer.amsgrad",
        type=bool,
        default=False,
    )

    parser.add_argument(
        "--lr_scheduler.monitor",
        type=str,
        default="val/loss",
    )
    parser.add_argument(
        "--lr_scheduler.interval",
        type=str,
        default="step",
    )
    parser.add_argument(
        "--lr_scheduler.warmup_epochs",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--lr_scheduler.T_max",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--lr_scheduler.warmup_start_lr",
        type=float,
        default=1e-9,
    )
    parser.add_argument(
        "--lr_scheduler.eta_min",
        type=float,
        default=1e-7,
    )

    parser.add_class_arguments(Trainer, "trainer", instantiate=False)

    return parser


def setup_env_and_args(cfg):
    assert cfg.backbone_cpt is not None
    assert cfg.num_mem_tokens is not None

    # set current working dir
    cfg.working_dir = str(Path(cfg.working_dir).expanduser().absolute())
    os.chdir(cfg.working_dir)
    if os.environ.get("LOCAL_RANK", 0) == 0:
        logger.info(f"Precision: {cfg.trainer.precision}")

    # Aydar # Pass memory settings to pretrained model
    if cfg.memory_forward_func is not None:
        cfg.memory_forward_func = get_cls_by_name(cfg.memory_forward_func)

    # Curriculum learning
    curriculum = {
        "n_epochs": cfg.curriculum[::2],
        "max_n_segments": cfg.curriculum[1::2],
    }
    cfg.curriculum = curriculum

    cfg.trainer.val_check_interval = cfg.trainer.val_check_interval // cfg.batch_size
    if cfg.trainer.log_every_n_steps is None:
        cfg.trainer.log_every_n_steps = 50
    cfg.trainer.log_every_n_steps = (
        cfg.trainer.log_every_n_steps #// cfg.trainer.accumulate_grad_batches
    )
    cfg.lr_scheduler.warmup_epochs = (
        cfg.lr_scheduler.warmup_epochs #// cfg.trainer.accumulate_grad_batches
    )

    pprint(cfg.as_dict())


def get_tokenizer(cfg):
    logger.info(f"Loading tokenizer from {cfg.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
    return tokenizer


def get_logger(cfg):
    Path(cfg.logger.save_dir).mkdir(parents=True, exist_ok=True)

    config = cfg.logger.as_dict()
    config["config"] = cfg.as_dict()
    config["id"] = cfg.logger.id if cfg.logger.resume else None

    wandb_logger = WandbLogger(**config)

    wandb_logger.LOGGER_JOIN_CHAR = "/"

    if os.environ.get("LOCAL_RANK", 0) == 0:
        logger.info(f"Logger run_id: {wandb_logger.version}")
        logger.info(f"Log_dir: {wandb_logger.save_dir}")

    return wandb_logger


def get_datasets(cfg, tokenizer):
    if cfg.task_name == "lm":
        data_cls = RMTDocsDataset
        split_list = ["train", "val"]
    elif cfg.task_name == "proofs":
        data_cls = RMTProofsDataset
        split_list = ["train", "val_test"]

    # get datasets
    segment_size = (
        cfg.input_size - 2 * cfg.num_mem_tokens - tokenizer.num_special_tokens_to_add()
    )
    if os.environ.get("LOCAL_RANK", 0) == 0:
        logger.info(f"preparing dataset for {cfg.task_name}")
        logger.info(f"segment_size = {segment_size}")

    datasets = {}
    data_dir = Path(cfg.data_dir)
    lemmas_path = Path(cfg.lemmas_path)
    for split in split_list:
        if (data_dir / f"{split}_tokenized.ckpt").exists():
            datasets[split] = data_cls.load_tokenized(
                data_dir / f"{split}_tokenized.ckpt"
            )
        else:
            datasets[split] = data_cls(data_dir / split, lemmas_path, tokenizer, cfg.max_n_segments,
                                      segment_length=segment_size,
                                      short_proofs_only=cfg.short_proofs_only,
                                      every_segment_def=cfg.every_segment_def,
                                      exclude_relevant_lemmas=cfg.exclude_relevant_lemmas,
                                      use_random_lemmas_names=cfg.use_random_lemmas_names)
            datasets[split].filter_shorts()
            datasets[split].tokenize()
            datasets[split].save_tokenized(str(data_dir / f"{split}_tokenized.ckpt"))

        if hasattr(datasets[split], "split_to_segments"):
            datasets[split].split_to_segments(segment_size)
        print(f"{split}: {len(datasets[split])}")

    if cfg.task_name == "proofs":
        datasets["val"] = datasets.pop("val_test")
        datasets["train"].segment_length = segment_size
        datasets["val"].segment_length = segment_size

    return datasets


def get_dataloaders(cfg, datasets):
    loader_cls = RMTDocsAllAtOnceDataLoader
    loader_kwargs = {
        "pin_memory": True,
        "num_workers": cfg.data_n_workers,
        "batch_size": cfg.batch_size,
        "drop_last": True,
    }

    loaders = {}

    loaders["train"] = loader_cls(
        datasets["train"],
        shuffle=True,
        **loader_kwargs,
    )

    loaders["val"] = loader_cls(
        datasets["val"],
        **loader_kwargs,
    )

    return loaders


def _get_backbone_model(cfg):
    # Load backbone model
    backbone_cls = get_cls_by_name(cfg.backbone_cls)
    if os.environ.get("LOCAL_RANK", 0) == 0:
        logger.info(f"Using model class: {backbone_cls}")
        logger.info(f"Loading model from {cfg.backbone_cpt}")

    try:
        backbone = backbone_cls.from_pretrained(cfg.backbone_cpt)
        logger.info(f"Model loaded from {cfg.backbone_cpt}")
    except OSError:
        backbone = backbone_cls(config=AutoConfig.from_pretrained(cfg.backbone_cpt))

    return backbone


def _get_pl_model(cfg, rmt_model):
    if cfg.pretrained_ckpt and not cfg.resume_training:
        logger.info(f"Loading checkpoint: {cfg.pretrained_ckpt}")
        pl_model = RMTModelPL.load_from_checkpoint(
            cfg.pretrained_ckpt, rmt_model=rmt_model
        )

        # TODO: update config properly
        pl_model.cfg.proof_loss_only = (
            cfg.proof_loss_only if hasattr(cfg, "proof_loss_only") else False
        )

        pl_model.save_hyperparameters(ignore=["rm_model"])
    else:
        pl_model = RMTModelPL(rmt_model, cfg)

    if cfg.freeze_model_weights:
        for n, p in pl_model.named_parameters():
            if "memory" not in n and "wte" not in n:
                p.requires_grad = False
        if os.environ.get("LOCAL_RANK", 0) == 0:
            logger.info(f"Frozen moodel weights except embeddings")

    # try:
    #     pl_model._module.model = torch.compile(pl_model._module.model)
    #     print("Model compiled")
    # except:
    #     pass
    return pl_model


def get_rmt_model(cfg, tokenizer):
    rmt_config = {
        "num_mem_tokens": cfg.num_mem_tokens,
        "max_n_segments": cfg.max_n_segments,
        "input_size": cfg.input_size,
        "bptt_depth": cfg.bptt_depth,
        "sum_loss": cfg.sum_loss,
        "def_lemmas_loss_weight": cfg.def_lemmas_loss_weight,
        "proof_loss_weight": cfg.proof_loss_weight,
        "tokenizer": tokenizer,
        "memory_forward_func": cfg.memory_forward_func,
        "memory_layers": cfg.memory_layers,
        "share_memory_layers": cfg.share_memory_layers,
        "reconstruction_loss_coef": cfg.reconstruction_loss_coef,
        "proof_loss_only": cfg.proof_loss_only
        if hasattr(cfg, "proof_loss_only")
        else False,
        "proofstep_token_id": tokenizer.vocab["[PROOFSTEP]"],
        "use_recur_mem": cfg.use_recur_mem,
    }

    backbone = _get_backbone_model(cfg)

    # Load RMT model
    rmt_cls = get_cls_by_name(cfg.rmt_cls)
    if os.environ.get("LOCAL_RANK", 0) == 0:
        logger.info(f"Wrapping in: {rmt_cls}")

    rmt_model = rmt_cls(backbone, **rmt_config)
    pl_model = _get_pl_model(cfg, rmt_model)

    return pl_model


def get_base_model(cfg, tokenizer):
    #if cfg.proof_loss_only:
    #    raise NotImplementedError
    backbone = _get_backbone_model(cfg)
    pl_model = _get_pl_model(cfg, backbone)
    # pl_model = torch.compile(pl_model)
    return pl_model


def get_trainer_callbacks(n_segments):
    callbacks = [
        ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_last=True,
            save_top_k=1,
            auto_insert_metric_name=False,
            filename=f"n_segments={n_segments}"
            + "-epoch={epoch:02d}-step={step}-loss={val/loss:.4f}",
        ),
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(
            monitor="val/loss",
            mode="min",
            strict=False,
            patience=5,
            check_finite=False,
        ),
    ]
    return callbacks

def generate_proof(model, batch, max_new_tokens=30):
    for i, segment in enumerate(batch):
        for key in segment:
            segment[key] = torch.stack([segment[key]]).to(model.device)
        segment['batch_idx'] = i
        
    lemmas = batch[:-1]
    proof = batch[-1]
    def_len = batch[0]["def_len"][0]
    proof["input_ids"][0][def_len + 1:] = tokenizer.pad_token_id
    proof["attention_mask"][0][def_len + 1:] = 0
    
    for idx in range(def_len + 1, def_len + 1 + max_new_tokens):
        model._module.reset_memory()
        for segment in lemmas:
            model(segment)
        logits = model(proof)["logits"][0][idx - 1]
        proof["input_ids"][0][idx] = torch.argmax(logits)
        proof["attention_mask"][0][idx] = 1
    
    return proof["input_ids"][0]
        
def generate_goal(model, batch, max_new_tokens=30):
    for i, segment in enumerate(batch):
        for key in segment:
            segment[key] = torch.stack([segment[key]]).to(model.device)
        segment['batch_idx'] = i
        
    lemmas = batch[:-1]
    prelast_seg = batch[-2]
    def_len = batch[0]["def_len"][0]
    prelast_seg["input_ids"][0][1:] = tokenizer.pad_token_id
    prelast_seg["attention_mask"][0][1:] = 0
    
    mem = None
    for idx in range(1, max_new_tokens):
        model._module.reset_memory()
        #for segment in lemmas:
        #    model(segment)
        #newmem = model._module.memory_states[-1][1]
#         if mem is not None:
#             print("MEMORY REL DIFF", torch.norm(mem - newmem) / torch.norm(mem))
#         mem = newmem
        logits = model(prelast_seg)["logits"][0][idx - 1]
        prelast_seg["input_ids"][0][idx] = torch.argmax(logits)
        prelast_seg["attention_mask"][0][idx] = 1
    
    return prelast_seg["input_ids"][0]

if __name__ == "__main__":
    parser = setup_parser()
    cfg = parser.parse_args()
    
    seed_everything(cfg.seed)

    setup_env_and_args(cfg)

    tokenizer = get_tokenizer(cfg)
    wandb_logger = get_logger(cfg)

    if cfg.model_type == "rmt":
        model = get_rmt_model(cfg, tokenizer)
    elif cfg.model_type == "base":
        model = get_base_model(cfg, tokenizer)

    datasets = get_datasets(cfg, tokenizer)

    # find resume ckpt path
    ckpt_dir = (
        Path(wandb_logger.save_dir)
        / wandb_logger._project
        / "1qcgogez" # TODO: make a parameter
        / "checkpoints"
    )
    resume_ckpt_path = ckpt_dir / "last-v5.ckpt"
    logger.info(f"Evaluation checkpoint: {resume_ckpt_path}")
    model = RMTModelPL.load_from_checkpoint(
        resume_ckpt_path, rmt_model=model._module
    )
    max_n_segments = 2
    for split, ds in datasets.items():
        ds.set_max_n_segments(max_n_segments)
    cfg.max_n_segments = max_n_segments

    if hasattr(model._module, "set_max_n_segments"):
        model._module.set_max_n_segments(max_n_segments)

    if hasattr(model._module, "reset_memory"):
        model._module.reset_memory()

    model.eval()
    
    max_new_tokens = 200
    
    
    for split in ["train", "val"]:
        print('-' * 100, f"Split: {split}", '-' * 100, sep='\n')
        dataset = datasets[split]
        for idx in range(0, len(dataset), len(dataset) // 11):
            print('-' * 100)
            print("Label:", tokenizer.decode(dataset[idx][-2]["input_ids"]), sep="\n")
            print("Genrated:", tokenizer.decode(generate_goal(model, dataset[idx], max_new_tokens)), sep="\n")
    
    for idx in range(0, 1000, 100):
        print('-' * 100)
        print("Label:", tokenizer.decode(datasets["val"][idx][-2]["input_ids"], skip_special_tokens=True), sep="\n")
        print("Genrated:", tokenizer.decode(generate_goal(model, datasets["val"][idx], max_new_tokens), skip_special_tokens=True), sep="\n")
    
#     for split in ["train", "val"]:
#         print('-' * 100, f"Split: {split}", '-' * 100, sep='\n')
#         dataset = datasets[split]
#         for idx in range(0, len(dataset), len(dataset) // 11):
#             print('-' * 100)
#             print("Label:", tokenizer.decode(dataset[idx][-1]["input_ids"], skip_special_tokens=True), sep="\n")
#             print("Genrated:", tokenizer.decode(generate_proof(model, dataset[idx], max_new_tokens), skip_special_tokens=True), sep="\n")
    
#     for idx in range(0, 1000, 100):
#         print('-' * 100)
#         print("Label:", tokenizer.decode(datasets["val"][idx][-1]["input_ids"], skip_special_tokens=True), sep="\n")
#         print("Genrated:", tokenizer.decode(generate_proof(model, datasets["val"][idx], max_new_tokens), skip_special_tokens=True), sep="\n")