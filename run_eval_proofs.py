import os
from pathlib import Path
from pprint import pprint

import torch
import tqdm
from jsonargparse import ArgumentParser

from lean_dataset import RMTDocsDataLoader
from modeling_rmt.lightning import RMTModelPL
from run_finetuning_lean_pl import (
    get_cls_by_name,
    get_datasets,
    get_tokenizer,
)


def get_best_ckpt_path(ckpt_dir, n_segments):
    best_ckpt, best_loss = None, float("inf")
    for path in Path(ckpt_dir).glob(f"n_segments={n_segments}*"):
        loss = float(path.name.split("loss=")[-1].split(".ckpt")[0])
        if loss < best_loss:
            best_loss = loss
            best_ckpt = path
    return best_ckpt


def load_cfg(args, ckpt_path):
    cfg = torch.load(
        Path(ckpt_path) / "checkpoint/mp_rank_00_model_states.pt", map_location="cpu"
    )["hyper_parameters"]["cfg"]
    # cfg.tokenizer = cfg.tokenizer.replace('../', args.working_dir)
    # cfg.backbone_cpt = cfg.backbone_cpt.replace('../', args.working_dir)
    # cfg.data_dir = cfg.data_dir.replace('../', args.workgin_dir)
    return cfg


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
    backbone = backbone_cls.from_pretrained(cfg.backbone_cpt)

    # Load RMT model
    rmt_cls = get_cls_by_name(cfg.rmt_cls)
    rmt_model = rmt_cls(backbone, **rmt_config)

    pl_model = RMTModelPL.load_from_checkpoint(cfg.pretrained_ckpt, rmt_model=rmt_model)

    try:
        pl_model = torch.compile(pl_model)
    except:
        pass

    return pl_model


def eval_model(cfg, model, tokenizer, datasets, max_n_segments, batch_size, num_workers):
    losses = []

    device = next(iter(model.parameters())).device

    with torch.no_grad():
        model.eval()
        model._module.reset_memory()
        datasets["val"].set_max_n_segments(max_n_segments)
        loader = RMTDocsDataLoader(
            datasets["val"], batch_size=batch_size, drop_last=True,
            num_workers=num_workers
        )

        model._module._actual_max_n_segments = max_n_segments
        model._module.reset_memory()
        for batch in tqdm.tqdm(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop('labels')
            out = model(batch)

            proofstep_idx = torch.where(
                batch["input_ids"] == tokenizer.vocab["[PROOFSTEP]"]
            )
            if not len(proofstep_idx[0]):
                continue

            lm_logits = out.logits[:, cfg.num_mem_tokens : -cfg.num_mem_tokens]

            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            for i, j in zip(*proofstep_idx):
                shift_labels[i.item()][: j.item() - 1] = -100

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

            losses.append(loss.cpu())

        losses = torch.stack(losses, dim=0)

    return losses


def main():

    parser = ArgumentParser()
    parser.add_argument('--run_id', type=str)
    parser.add_argument('--n_segments_train', type=int, nargs="+")
    parser.add_argument('--n_segments_val', type=int, nargs="+")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--working_dir', type=str, default='.')

    args = parser.parse_args()

    tokenizer = None
    datasets = None

    device = "cuda"

    working_dir = str(Path(args.working_dir).expanduser().absolute())
    os.chdir(working_dir)
    args.working_dir = working_dir

    pprint(args.as_dict())

    ckpt_dir = str(Path(working_dir) / f"logs/rmt_proofs/{args.run_id}/checkpoints/")
    print(f"Run id: {args.run_id}")

    n_segments_train_list = args.n_segments_train
    n_segments_val_list = args.n_segments_val

    for n_segments_train in n_segments_train_list:

        print(f"\tn_segments_train: {n_segments_train}")
        print(ckpt_dir)
        best_ckpt = get_best_ckpt_path(ckpt_dir, n_segments=n_segments_train)

        if best_ckpt is None:
            return

        cfg = load_cfg(args, best_ckpt)
        cfg.pretrained_ckpt = best_ckpt

        pprint(cfg.as_dict())

        if tokenizer is None:
            tokenizer = get_tokenizer(cfg)
            datasets = get_datasets(cfg, tokenizer)

        model = get_model(cfg, tokenizer)
        model.to(device)


        for n_segments_val in n_segments_val_list:
            total_loss = eval_model(
                cfg, model, tokenizer, datasets, max_n_segments=n_segments_val, batch_size=args.batch_size,
                num_workers=args.num_workers
            )

            mean_loss = total_loss[n_segments_val - 1 :: n_segments_val].mean()
            perplexity = torch.exp(mean_loss)
            print(
                f"\ttrain_segments: {n_segments_train}, val_segments: {n_segments_val}, loss: {mean_loss:.3f}, perplexity: {perplexity:.3f}"
            )

            torch.save(
                total_loss,
                str(Path(working_dir) / f"logs/proof_losses_{args.run_id}_{n_segments_train}_{n_segments_val}.ckpt")
            )


if __name__ == "__main__":
    main()