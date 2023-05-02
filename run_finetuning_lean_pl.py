import importlib
import logging
import os
from pathlib import Path
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
from transformers import AutoTokenizer  # noqa: E402

from lean_dataset import RMTDocsDataLoader, RMTDocsDataset
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
    parser.add_argument("--task_name", type=str, help="Task name, wikitext, ...")
    parser.add_argument("--data_dir", type=str, help="Path to the data directory")
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
        "Example: `--curriculum 1 1 1 2 1 5`. "
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
        cfg.trainer.log_every_n_steps // cfg.trainer.accumulate_grad_batches
    )
    cfg.lr_scheduler.warmup_epochs = (
        cfg.lr_scheduler.warmup_epochs // cfg.trainer.accumulate_grad_batches
    )


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
    # get datasets
    segment_size = (
        cfg.input_size - 2 * cfg.num_mem_tokens - tokenizer.num_special_tokens_to_add()
    )
    if os.environ.get("LOCAL_RANK", 0) == 0:
        logger.info(f"preparing dataset for {cfg.task_name}")
        logger.info(f"segment_size = {segment_size}")

    datasets = {}
    data_dir = Path(cfg.data_dir)
    for split in ["train", "val"]:
        if (data_dir / f"{split}_tokenized.ckpt").exists():
            datasets[split] = RMTDocsDataset.load_tokenized(
                data_dir / f"{split}_tokenized.ckpt"
            )
        else:
            datasets[split] = RMTDocsDataset(
                data_dir / split, tokenizer, cfg.max_n_segments
            )
            datasets[split].tokenize()
            datasets[split].save_tokenized(str(data_dir / f"{split}_tokenized.ckpt"))
        datasets[split].split_to_segments(segment_size)

    return datasets


def get_dataloaders(cfg, datasets):
    loader_kwargs = {
        "pin_memory": True,
        "num_workers": cfg.data_n_workers,
        "batch_size": cfg.batch_size,
        # * cfg.gradient_accumulation_steps,  # batch size per GPU
        "drop_last": True,
    }

    loaders = {}

    loaders["train"] = RMTDocsDataLoader(
        datasets["train"],
        shuffle=True,
        **loader_kwargs,
    )

    loaders["val"] = RMTDocsDataLoader(
        datasets["val"],
        **loader_kwargs,
    )

    return loaders


def get_model(cfg, tokenizer):
    rmt_config = {
        "num_mem_tokens": cfg.num_mem_tokens,
        "max_n_segments": cfg.max_n_segments,
        "input_size": cfg.input_size,
        "bptt_depth": cfg.bptt_depth,
        "sum_loss": cfg.sum_loss,
        "tokenizer": tokenizer,
        "memory_forward_func": cfg.memory_forward_func,
        "memory_layers": cfg.memory_layers,
        "share_memory_layers": cfg.share_memory_layers,
        "reconstruction_loss_coef": cfg.reconstruction_loss_coef,
    }

    # Load backbone model
    backbone_cls = get_cls_by_name(cfg.backbone_cls)
    if os.environ.get("LOCAL_RANK", 0) == 0:
        logger.info(f"Using model class: {backbone_cls}")

    backbone = backbone_cls.from_pretrained(cfg.backbone_cpt)

    # Load RMT model
    rmt_cls = get_cls_by_name(cfg.rmt_cls)
    if os.environ.get("LOCAL_RANK", 0) == 0:
        logger.info(f"Wrapping in: {rmt_cls}")

    rmt_model = rmt_cls(backbone, **rmt_config)

    if cfg.pretrained_ckpt:
        logger.info(f"Loading checkpoint: {cfg.pretrained_ckpt}")
        pl_model = RMTModelPL.load_from_checkpoint(
            cfg.pretrained_ckpt, rmt_model=rmt_model
        )
        pl_model.save_hyperparameters(ignore=['rm_model'])
    else:
        pl_model = RMTModelPL(rmt_model, cfg)

    if cfg.freeze_model_weights:
        for n, p in pl_model.named_parameters():
            if "memory" not in n and "wte" not in n:
                p.requires_grad = False
        if os.environ.get("LOCAL_RANK", 0) == 0:
            logger.info(f"Frozen moodel weights except embeddings")

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
            patience=3,
            check_finite=False,
        ),
    ]
    return callbacks


if __name__ == "__main__":
    parser = setup_parser()
    cfg = parser.parse_args()

    seed_everything(cfg.seed)

    setup_env_and_args(cfg)

    tokenizer = get_tokenizer(cfg)
    wandb_logger = get_logger(cfg)

    model = get_model(cfg, tokenizer)

    datasets = get_datasets(cfg, tokenizer)

    # find resume ckpt path
    max_n_segments_ckpt = 0
    resume_ckpt_path = None
    if cfg.resume_training:
        ckpt_dir = (
            Path(wandb_logger.save_dir)
            / wandb_logger._project
            / wandb_logger.version
            / "checkpoints"
        )
        resume_ckpt_path = str(ckpt_dir / "last.ckpt")

        for path in ckpt_dir.glob("*.ckpt"):
            if "n_segments=" in path.name:
                n_segments = path.name.split("n_segments=")[1]
                n_segments = int(n_segments.split("-epoch")[0])
                max_n_segments_ckpt = max(max_n_segments_ckpt, n_segments)

        logger.info(f"Resuming from: {resume_ckpt_path}")
        logger.info(f"Resuming n_segments: {max_n_segments_ckpt}")

    for n_epochs, max_n_segments in zip(
        cfg.curriculum["n_epochs"], cfg.curriculum["max_n_segments"]
    ):
        if max_n_segments < max_n_segments_ckpt:
            continue

        for split, ds in datasets.items():
            ds.set_max_n_segments(max_n_segments)
        loaders = get_dataloaders(cfg, datasets)

        cfg.max_n_segments = max_n_segments
        cfg.trainer.max_steps = n_epochs * len(loaders["train"])
        model.cfg.lr_scheduler.T_max = (
            cfg.trainer.max_steps // cfg.trainer.accumulate_grad_batches
        )
        cfg.trainer.max_epochs = None
        model._module.set_max_n_segments(max_n_segments)
        wandb_logger._prefix = f"seg_len-{max_n_segments}"

        trainer_options = cfg.trainer.as_dict()
        trainer_options["logger"] = wandb_logger
        trainer_options["callbacks"] = get_trainer_callbacks(max_n_segments)
        trainer = Trainer(**trainer_options)

        logger.info("-" * 80)
        logger.info(f"Max number of segments: {max_n_segments}")
        logger.info(f'N batches per epoch: {len(loaders["train"])}')
        logger.info(f"Trainer max epochs: {trainer.max_epochs}")

        model._module.reset_memory()

        if cfg.validate_only:
            trainer.validate(model, loaders["val"])
        elif max_n_segments == max_n_segments_ckpt:
            trainer.fit(
                model,
                train_dataloaders=loaders["train"],
                val_dataloaders=loaders["val"],
                ckpt_path=resume_ckpt_path,
            )
        else:
            trainer.fit(
                model,
                train_dataloaders=loaders["train"],
                val_dataloaders=loaders["val"],
            )
