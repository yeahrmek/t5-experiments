import os
import random
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
import tqdm
from jsonargparse import ArgumentParser
from pytorch_lightning import seed_everything

from lean_dataset import RMTDocsDataLoader
from run_finetuning_lean_pl import (
    get_base_model,
    get_datasets,
    get_rmt_model,
    get_tokenizer,
)

max_seed_value = np.iinfo(np.uint32).max
min_seed_value = np.iinfo(np.uint32).min


def get_best_ckpt_path(ckpt_dir, n_segments):
    best_ckpt, best_loss = None, float("inf")
    for path in Path(ckpt_dir).glob(f"n_segments={n_segments}*"):
        loss = float(path.name.split("loss=")[-1].split(".ckpt")[0])
        if loss < best_loss:
            best_loss = loss
            best_ckpt = path

    if not best_ckpt:
        if n_segments == 1:
            best_ckpt = Path(ckpt_dir) / "last.ckpt"
        else:
            best_ckpt = Path(ckpt_dir) / f"last-v{n_segments - 1}.ckpt"

    if not best_ckpt.exists():
        return None

    return best_ckpt


def load_cfg(args, ckpt_path):
    cfg = torch.load(
        Path(ckpt_path) / "checkpoint/mp_rank_00_model_states.pt", map_location="cpu"
    )["hyper_parameters"]["cfg"]
    # cfg.tokenizer = cfg.tokenizer.replace('../', args.working_dir)
    # cfg.backbone_cpt = cfg.backbone_cpt.replace('../', args.working_dir)
    # cfg.data_dir = cfg.data_dir.replace('../', args.workgin_dir)
    return cfg


def eval_model(
    cfg, model, tokenizer, datasets, max_n_segments, batch_size, num_workers
):
    assert batch_size == 1
    losses = []
    proofstep_indices = []
    pad_start_indices = []
    batch_indices = []

    device = next(iter(model.parameters())).device

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        model.eval()
        datasets["val"].set_max_n_segments(max_n_segments)
        loader = RMTDocsDataLoader(
            datasets["val"],
            batch_size=batch_size,
            num_workers=num_workers,
        )

        if hasattr(model._module, "reset_memory"):
            model._module._actual_max_n_segments = max_n_segments
            model._module.reset_memory()
        loss_sum = 0
        n_elements = 0
        with tqdm.tqdm(loader) as pbar:
            for batch in pbar:
                batch = {k: v.to(device) if k != 'batch_idx' else v for k, v in batch.items()}
                labels = batch.pop("labels")
                out = model(batch)

                lm_logits = out.logits
                if not hasattr(cfg, "model_type") or cfg.model_type == "rmt":
                    lm_logits = out.logits[:, cfg.num_mem_tokens : -cfg.num_mem_tokens]

                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )

                attn_mask = batch.get('attention_mask', None)
                if attn_mask is not None:
                    batch = {k: v[0][attn_mask[0] == 1].reshape(1, -1) if k != 'batch_idx' else v for k, v in batch.items()}
                    loss = loss[attn_mask[0][1:] == 1]

                proofstep_idx = torch.where(
                    batch["input_ids"] == tokenizer.vocab["[PROOFSTEP]"]
                )

                losses.append(loss.cpu())

                if len(proofstep_idx[0]) > 0:
                    proofstep_idx = proofstep_idx[1].item()
                else:
                    proofstep_idx = len(loss)

                proofstep_indices.append(proofstep_idx)
                pad_start_indices.append(batch["attention_mask"].sum().item())
                batch_indices.append(batch['batch_idx'])

                loss_sum += loss.sum()
                n_elements += len(loss)
                pbar.set_postfix(loss=loss_sum / n_elements)
                pbar.update(1)

        # losses = torch.stack(losses, dim=0)

    return losses, proofstep_indices, pad_start_indices, batch_indices


def main():
    parser = ArgumentParser()
    parser.add_argument("--run_id", type=str, nargs="+")
    parser.add_argument("--n_segments_train", type=int, nargs="+")
    parser.add_argument("--n_segments_val", type=int, nargs="+")
    parser.add_argument("--padding_side", type=str, default="right")
    parser.add_argument("--add_lemma_token", type=bool, default=False)
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--working_dir", type=str, default=".")
    parser.add_argument("--proof_only_loss", type=bool, default=False)

    args = parser.parse_args()

    tokenizer = None
    datasets = None

    device = "cuda"

    working_dir = str(Path(args.working_dir).expanduser().absolute())
    os.chdir(working_dir)
    args.working_dir = working_dir

    pprint(args.as_dict())

    n_segments_train_list = args.n_segments_train
    n_segments_val_list = args.n_segments_val
    run_id_list = args.run_id

    for run_id in run_id_list:
        if args.proof_only_loss:
            ckpt_dir = str(Path(working_dir) / f"logs/rmt_proofs_only/{run_id}/checkpoints/")
        else:
            ckpt_dir = str(Path(working_dir) / f"logs/rmt_proofs/{run_id}/checkpoints/")
        print(f"Run id: {run_id}")

        save_filename_prefix = ""
        for n_segments_train in n_segments_train_list:
            print(f"\tn_segments_train: {n_segments_train}")
            print(ckpt_dir)
            best_ckpt = get_best_ckpt_path(ckpt_dir, n_segments=n_segments_train)

            if best_ckpt is None:
                continue

            cfg = load_cfg(args, best_ckpt)
            cfg.pretrained_ckpt = best_ckpt

            if tokenizer is None:
                tokenizer = get_tokenizer(cfg)
                datasets = get_datasets(cfg, tokenizer)
                for key, ds in datasets.items():
                    ds.padding_side = args.padding_side
                    ds.add_lemma_token = args.add_lemma_token

            if not hasattr(cfg, "model_type") or cfg.model_type == "rmt":
                model = get_rmt_model(cfg, tokenizer)
            elif cfg.model_type == "base":
                model = get_base_model(cfg, tokenizer)
                model._module.resize_token_embeddings(50304)


            model.to(device)
            if hasattr(model._module, 'rmt_config'):
                model._module.rmt_config['padding_side'] = args.padding_side

            for n_segments_val in n_segments_val_list:
                for run in range(args.n_runs):
                    seed = random.randint(min_seed_value, max_seed_value)  # noqa: S311
                    seed = seed_everything(seed)
                    print(f"Using random seed: {seed}")
                    total_loss, proofstep_idx, mask_sum, batch_idx = eval_model(
                        cfg,
                        model,
                        tokenizer,
                        datasets,
                        max_n_segments=n_segments_val,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                    )

                    print(
                        f"\ttrain_segments: {n_segments_train}, val_segments: {n_segments_val}"#, loss: {mean_loss.mean():.3f}, perplexity: {perplexity:.3f}"
                    )

                    suffix = 'lemmatok' if args.add_lemma_token else 'nolemmatok'
                    torch.save(
                        {
                            "loss": total_loss,
                            "proofstep_idx": proofstep_idx,
                            "mask_sum": mask_sum,
                            'batch_idx': batch_idx
                        },
                        str(
                            Path(working_dir)
                            / (
                                f"logs/validation_correct/{save_filename_prefix}losses_"
                                f"{run_id}_"
                                f"{args.padding_side}_"
                                f"{n_segments_train}_"
                                f"{n_segments_val}_{seed}{suffix}.ckpt"
                            )
                        ),
                    )


if __name__ == "__main__":
    main()
