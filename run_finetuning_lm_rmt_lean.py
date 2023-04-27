import json
import logging
import math
import os
from pathlib import Path

import horovod.torch as hvd
import torch
import transformers  # noqa: E402
from dotenv import load_dotenv
from torch.utils.data import DistributedSampler
from transformers import AutoTokenizer  # noqa: E402
from transformers import HfArgumentParser
from pytorch_lightning.loggers import WandbLogger

import lm_experiments_tools.optimizers as optimizers  # noqa: E402
from lm_experiments_tools.lean_dataset import RMTDocsDataLoader, RMTDocsDataset
from lm_experiments_tools import TrainerArgs
from lm_experiments_tools.trainer_tbptt import Trainer
from lm_experiments_tools.utils import collect_run_configuration, get_cls_by_name
from lm_experiments_tools.utils import get_optimizer as get_optimizer_cls  # noqa: E402

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# if CUDA_VISIBLE_DEVICES is not set make all gpus visible
if os.environ.get("CUDA_VISIBLE_DEVICES", None) is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        [str(i) for i in range(torch.cuda.device_count())]
    )

logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
# first call to torch.cuda.device_count() sets visible gpus, following calls will not change the result
logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")

hvd.init()


# limit # of CPU threads to be used per pytorch worker, otherwise it might use all cpus and throttle gpus
# > 2 fails cause of https://github.com/pytorch/pytorch/issues/56615
# need to upgrade to torch>1.8.1
torch.set_num_threads(4)
# all gpus set with CUDA_VISIBLE_DEVICES are visible to process, indexing from 0 to ...
torch.cuda.set_device(hvd.local_rank())


def setup_parser():
    parser = HfArgumentParser(TrainerArgs)
    parser.add_argument("--task_name", type=str, help="Task name, wikitext, ...")
    parser.add_argument("--data_dir", type=str, help="Path to the data directory")
    parser.add_argument("--log_dir", type=str, help="Path to the log directory")
    parser.add_argument("--wandb_project", type=str, help="Name of wandb project")
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
        default=2,
        help="number of dataloader workers (default: 2)",
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
        "--rmt_cpt", type=str, default=None, help="pretrained model checkpoint path"
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
        type=int,
        nargs="+",
        help="Scheduler for number of segments to train on. "
        "Input should be in the following format: <n_iters> <n_segments> <n_iters> <n_segments> ..."
        "Example: `--curriculum 1000 1 1000 2 2000 5`. "
        "In this example we will first train for 1000 iters on 1 segment "
        "then train for 1000 iters on 2 segments, then for 2000 iters on 5 segments",
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
        "--optimizer",
        type=str,
        default="AdamW",
        help="optimizer name: AdamW, Adafactor. (default: AdamW)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="optimizer weight decay (default: 0.0)",
    )
    parser.add_argument(
        "--scale_parameter",
        action="store_true",
        default=False,
        help="Adafactor scale_parameter (default: False)",
    )
    parser.add_argument(
        "--relative_step",
        action="store_true",
        default=False,
        help="Adafactor relative_step (default: False)",
    )
    parser.add_argument(
        "--warmup_init",
        action="store_true",
        default=False,
        help="Adafactor warmup_init (default: False)",
    )
    return parser


def setup_env_and_args(args):
    assert args.backbone_cpt is not None
    assert args.num_mem_tokens is not None

    # set current working dir
    args.working_dir = str(Path(args.working_dir).expanduser().absolute())
    os.chdir(args.working_dir)
    if hvd.rank() == 0:
        logger.info(f"hvd size: {hvd.size()}")
        logger.info(f"FP16: {args.fp16}")

    if args.valid_interval is None:
        args.valid_interval = args.log_interval

    # Aydar # Pass memory settings to pretrained model
    if args.memory_forward_func is not None:
        args.memory_forward_func = get_cls_by_name(args.memory_forward_func)

    # Curriculum learning
    curriculum = {
        "n_iters": args.curriculum[::2],
        "max_n_segments": args.curriculum[1::2],
    }
    args.curriculum = curriculum


def get_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    return tokenizer


def get_logger(args):

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    # get absolute path
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        save_dir=args.log_dir
    )
    wandb_logger.LOGGER_JOIN_CHAR = '/'
    args.logger_version = wandb_logger.version
    args.model_path = str(Path(args.log_dir) / wandb_logger.version / 'checkpoints')

    if hvd.rank() == 0:
        logger.info(f"Logger run_id: {wandb_logger.version}")
        logger.info(f"Log_dir: {wandb_logger.save_dir}")
        logger.info(f"Model checkpoint path: {args.model_path}")

    return wandb_logger


def get_datasets(args, tokenizer):
    # get datasets
    segment_size = args.input_size - 2 * args.num_mem_tokens - tokenizer.num_special_tokens_to_add()
    if hvd.rank() == 0:
        logger.info(f"preparing dataset for {args.task_name}")
        logger.info(f"segment_size = {segment_size}")

    datasets = {}
    for split in ["train", "val"]:
        datasets[split] = RMTDocsDataset(
            Path(args.data_dir) / split, tokenizer, args.max_n_segments
        )
        datasets[split].tokenize()
        datasets[split].split_to_segments(segment_size)

    return datasets


def get_dataloaders(args, datasets):
    loader_kwargs = {
        "pin_memory": True,
        "num_workers": args.data_n_workers,
        "batch_size": args.batch_size
        * args.gradient_accumulation_steps,  # batch size per GPU
        "drop_last": True
    }

    loaders, samplers = {}, {}

    # shuffle train data each epoch (one loop over train_dataset)
    samplers["train"] = DistributedSampler(
        datasets["train"],
        rank=hvd.rank(),
        num_replicas=hvd.size(),
        shuffle=False,
        drop_last=True,
        seed=args.seed,
    )
    loaders["train"] = RMTDocsDataLoader(
        datasets["train"],
        sampler=samplers["train"],
        **loader_kwargs,
    )

    samplers["val"] = DistributedSampler(
        datasets["val"],
        rank=hvd.rank(),
        num_replicas=hvd.size(),
        shuffle=False,
        drop_last=True,
    )
    loaders["val"] = RMTDocsDataLoader(
        datasets["val"],
        sampler=samplers["val"],
        **loader_kwargs,
    )

    return loaders, samplers


def get_model(args, tokenizer):
    rmt_config = {
        "num_mem_tokens": args.num_mem_tokens,
        "max_n_segments": args.max_n_segments,
        "input_size": args.input_size,
        "bptt_depth": args.bptt_depth,
        "sum_loss": args.sum_loss,
        "tokenizer": tokenizer,
        "memory_forward_func": args.memory_forward_func,
        "memory_layers": args.memory_layers,
        "share_memory_layers": args.share_memory_layers,
        "reconstruction_loss_coef": args.reconstruction_loss_coef,
    }

    if hvd.rank() == 0 and args.log_dir is None:
        logger.warning(
            "model_path is not set: config, logs and checkpoints will not be saved."
        )

    # create model path and save configuration
    if hvd.rank() == 0 and args.model_path is not None:
        model_path = Path(args.model_path)
        if not model_path.exists():
            Path(model_path).mkdir(parents=True)
        args_dict = collect_run_configuration(args)
        # todo: if model path exists and there is config file, write new config file aside
        json.dump(args_dict, open(model_path / "config.json", "w"), indent=4)

    # Load backbone model
    backbone_cls = get_cls_by_name(args.backbone_cls)
    if hvd.rank() == 0:
        logger.info(f"Using model class: {backbone_cls}")

    backbone = backbone_cls.from_pretrained(args.backbone_cpt)

    # Load RMT model
    rmt_cls = get_cls_by_name(args.rmt_cls)
    if hvd.rank() == 0:
        logger.info(f"Wrapping in: {rmt_cls}")

    rmt_model = rmt_cls(backbone, **rmt_config)

    ## load cpt of rmt
    if args.rmt_cpt:
        rmt_cpt = os.path.join(args.rmt_cpt, "model_best.pth")
        state_dict = torch.load(rmt_cpt, map_location="cpu")
        rmt_model.load_state_dict(state_dict["model_state_dict"])
        if hvd.rank() == 0:
            logger.info(f"Loaded RMT state dict from: {args.rmt_cpt}")

    if args.freeze_model_weights:
        for n, p in rmt_model.named_parameters():
            if "memory" not in n and "wte" not in n:
                p.requires_grad = False
        if hvd.rank() == 0:
            logger.info(f"Frozen moodel weights except embeddings")

    return rmt_model


def get_optimizer(args, model):
    # define optimizer
    optimizer_cls = get_optimizer_cls(args.optimizer)
    if optimizer_cls is None:
        raise RuntimeError(
            f"{args.optimizer} was not found in optimizers, torch.optim, transformers.optimization"
        )

    if hvd.rank() == 0:
        logger.info(f"Using optimizer class: {optimizer_cls}")

    # todo: group optimizer params
    if optimizer_cls in [transformers.optimization.Adafactor, optimizers.Adafactor]:
        # https://github.com/huggingface/transformers/pull/9751/files -> transformers 4.3.0
        optimizer = optimizer_cls(
            model.parameters(),
            lr=args.lr,
            scale_parameter=args.scale_parameter,
            relative_step=args.relative_step,
            warmup_init=args.warmup_init,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = optimizer_cls(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    return optimizer


def train(trainer):
    # train loop
    trainer.train()
    # make sure all workers are done
    hvd.barrier()
    # run validation after training
    if args.save_best:
        best_model_path = str(Path(args.model_path) / "model_best.pth")
        if hvd.rank() == 0:
            logger.info(f"Loading best saved model from {best_model_path}")
        trainer.load(best_model_path)


def validate(trainer, loaders, split):
    # run validation, do not write to tensorboard
    if hvd.rank() == 0:
        logger.info(f"Running validation on {split} set:")
    if loaders[split] is not None:
        trainer.validate(loaders[split], split=split, write_tb=False)


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()

    setup_env_and_args(args)

    tokenizer = get_tokenizer(args)
    wandb_logger = get_logger(args)

    model = get_model(args, tokenizer)

    optimizer = get_optimizer(args, model)

    # for encoder only classification
    def keep_for_metrics_fn(batch, output):
        # select data from batch and model output that would be used to compute metrics
        data = {}
        data["labels"] = batch["labels"]
        data["loss"] = output["loss"]
        data["predictions"] = torch.argmax(output["logits"].detach(), dim=-1)
        return data

    # HF datasets can compute metrics on each gpu process and then aggregate them on process with rank 0
    # synchronization is done by using temporay files on a shared filesystem
    # rank and number of workers is set by num_process and process_id params
    # BUT our Trainer aggregates all prediction from all gpus!
    #   this will lead to computing metrics for predictions repeated xN_GPUS times
    # need to try:
    # - keep_in_memory=True, may lead to OOM for large validation sets, after sync predictions and targets for the full
    #       validation set would be stored on each GPU -> xN_GPUs RAM
    #   - implemented currently
    # - compute metrics on batch lvl
    # - add support of HF metrics and turn off aggregation in case if metric has .add_batch method
    # scrolls_metric = datasets.load_metric(scrolls_metric_path, args.task_name, keep_in_memory=True)

    def metrics_fn(data):
        # compute metrics based on stored labels, predictions, ...
        metrics = {}
        y, p = data["labels"], data["predictions"]
        if hvd.rank() == 0 and args.show_valid_examples > 0:
            for i in range(min(args.show_valid_examples, len(y))):
                logger.info(f"y: {y[i]}")
                logger.info(f"p: {p[i]}")
                logger.info("-" * 50)
        try:
            perplexity = math.exp(data["loss"].mean())
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        return metrics

    ### booydar
    batch_metrics_fn = lambda _, y: {
        key: y[key] for key in y.keys() if (("loss" in key) or ("!log" in key))
    }

    datasets = get_datasets(args, tokenizer)
    args.valid_interval = args.valid_interval // args.gradient_accumulation_steps
    args.num_warmup_steps = args.num_warmup_steps // args.gradient_accumulation_steps

    for n_iters, max_n_segments in zip(
        args.curriculum["n_iters"], args.curriculum["max_n_segments"]
    ):
        args.max_n_segments = max_n_segments
        model.rmt_config['max_n_segments'] = max_n_segments

        # multiply by max_n_segments because dataloader yields each segment
        # so trainer will consider each segment as a separate iteration
        args.iters = n_iters * max_n_segments // args.gradient_accumulation_steps

        wandb_logger._prefix = f"seg_len-{max_n_segments}"
        args.model_path = Path(args.model_path).parent / f"seg_len-{max_n_segments}" / "checkpoints"

        for split, ds in datasets.items():
            ds.set_max_n_segments(max_n_segments)
        loaders, samplers = get_dataloaders(args, datasets)

        trainer = Trainer(
            args,
            model,
            optimizer,
            loaders["train"],
            loaders["val"],
            samplers["train"],
            keep_for_metrics_fn=keep_for_metrics_fn,
            metrics_fn=metrics_fn,
            batch_metrics_fn=batch_metrics_fn,
            generate_kwargs={},
            wandb_logger=wandb_logger
        )

        if not args.validate_only:
            model.reset_memory()
            train(trainer)
        else:
            validate(trainer, loaders, "val")
