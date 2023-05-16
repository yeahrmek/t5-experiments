import json
import math
from itertools import chain
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer


class RMTDocsDataset:
    def __init__(
        self,
        data_dir: str,
        tokenizer: PreTrainedTokenizer,
        max_n_segments: int = -1,
        drop_last: bool = True,
    ):
        self.data_dir = str(Path(data_dir).resolve())
        self.tokenizer = tokenizer
        self.drop_last = drop_last
        self._max_n_segments = max_n_segments
        self._segment_len = None

        dataset = Dataset.from_parquet(
            [str(x) for x in Path(data_dir).glob("*.parquet")]
        )
        self.documents = dataset["input_ids"]

        self.input_ids = []
        self.attn_masks = []
        self._sequence_id = []
        self.batch_indices = []

    def __len__(self) -> int:
        if self._max_n_segments > 0:
            return len(self.batch_indices)
        return None

    def set_max_n_segments(self, max_n_segments: int) -> None:
        self._max_n_segments = max_n_segments
        self._calc_batch_indices()

    @property
    def n_segments(self) -> int:
        """
        Total number of segments
        """
        return len(self.input_ids)

    def tokenize(self) -> None:
        self.tokenized_documents = [
            self.tokenizer(
                doc + " " + self.tokenizer.eos_token,
                truncation=False,
                padding=False,
                return_tensors="pt",
            )["input_ids"].flatten()
            for doc in self.documents
        ]

    def save_tokenized(self, path: str) -> None:
        torch.save(
            {
                "tokenized_documents": self.tokenized_documents,
                "data_dir": self.data_dir,
                "tokenizer": self.tokenizer,
                "drop_last": self.drop_last,
                "max_n_segments": self._max_n_segments,
            },
            path,
        )

    @classmethod
    def load_tokenized(cls, path: str) -> "RMTDocsDataset":
        state_dict = torch.load(path)
        dataset = cls(
            state_dict["data_dir"],
            state_dict["tokenizer"],
            state_dict["max_n_segments"],
            state_dict["drop_last"],
        )
        dataset.tokenized_documents = state_dict["tokenized_documents"]
        return dataset

    def split_to_segments(self, segment_len: int) -> None:
        self._segment_len = segment_len
        self.input_ids = []
        self.attn_masks = []
        self._sequence_id = []
        for i, tensor in enumerate(self.tokenized_documents):
            n_segments = math.ceil(len(tensor) / segment_len)

            if n_segments == 0:
                continue

            # drop last segment if it contains only one token --- <eos_token>
            if len(tensor) % segment_len == 1:
                n_segments = len(tensor) // segment_len

            padded_tensor = torch.LongTensor(n_segments * segment_len).fill_(
                self.tokenizer.pad_token_id
            )
            seq_len = min(len(padded_tensor), len(tensor))
            padded_tensor[:seq_len] = tensor[:seq_len]

            chunks = list(torch.chunk(padded_tensor, n_segments, dim=0))

            attn_masks = [(ch != self.tokenizer.pad_token_id).long() for ch in chunks]

            self.input_ids.extend(chunks)
            self._sequence_id.extend([i] * len(chunks))
            self.attn_masks.extend(attn_masks)
        self._sequence_id = torch.LongTensor(self._sequence_id)

        self._calc_batch_indices()

    def _calc_batch_indices(self):
        """
        Create array of segment indices that can be used in __getitem__
        """
        self.batch_indices = []
        total_segments = 0
        for i, tensor in enumerate(self.tokenized_documents):
            n_segments = math.ceil(len(tensor) / self._segment_len)

            if n_segments == 0:
                continue

            # always drop last
            if n_segments >= self._max_n_segments:
                indices = torch.arange(
                    total_segments,
                    total_segments + n_segments - self._max_n_segments + 1,
                )
            else:
                indices = torch.LongTensor()

            self.batch_indices.append(indices)
            total_segments += n_segments
        self.batch_indices = torch.concat(self.batch_indices)

    def __getitem__(self, index: int) -> List[Dict[str, torch.Tensor]]:
        """
        Return a list of `max_n_segments` segments with attention masks
        """
        assert self._max_n_segments != -1

        idx = self.batch_indices[index]
        input_ids = self.input_ids[idx : idx + self._max_n_segments]
        attn_mask = self.attn_masks[idx : idx + self._max_n_segments]

        output = []
        for ids, mask in zip(input_ids, attn_mask):
            labels = ids.clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            output.append(
                {
                    "input_ids": ids,
                    "attention_mask": mask,
                    "labels": labels,
                }
            )

        return output


class RMTProofsDataset:
    def __init__(
        self,
        data_dir: str,
        lemmas_path: str,
        tokenizer: PreTrainedTokenizer,
        max_n_segments: int = -1,
        segment_length: int = 0,
        padding_side: str = 'right',
        add_lemma_token: bool = False
    ):
        assert padding_side in ['left', 'right']
        self.data_dir = str(Path(data_dir).resolve())
        self.lemmas_path = str(Path(lemmas_path).resolve())
        self.tokenizer = tokenizer
        self._max_n_segments = max_n_segments
        self.segment_length = segment_length
        self.padding_side = padding_side
        self.add_lemma_token = add_lemma_token

        self.dataset = Dataset.from_parquet(
            [str(x) for x in Path(data_dir).glob("*.parquet")]
        )

        with open(lemmas_path) as fh:
            self.lemmas = json.load(fh)

        self.tokenized = {}

    def __len__(self) -> int:
        return len(self.tokenized.get("decl_def", []))

    def set_max_n_segments(self, max_n_segments: int) -> None:
        self._max_n_segments = max_n_segments

    def tokenize(self) -> None:
        self.tokenized["decl_def"] = self.tokenizer(
            self.dataset["decl_def"],
            truncation=False,
            padding=False,
        )["input_ids"]

        self.tokenized["proof"] = self.tokenizer(
            [f"[PROOFSTEP] {x}" for x in self.dataset["full_proof"]],
            truncation=False,
            padding=False,
        )["input_ids"]

        args_defs_tokenized = []
        for doc in self.dataset:
            args_defs = [
                f"<lemma> {x.strip()}" for x in doc["args_defs"].split("<lemma>") if x.strip()
            ]
            if args_defs:
                args_defs_tokenized.append(
                    self.tokenizer(args_defs, truncation=False, padding=False)[
                        "input_ids"
                    ]
                )
            else:
                args_defs_tokenized.append([])

        self.tokenized["args"] = args_defs_tokenized

    def save_tokenized(self, path: str) -> None:
        torch.save(
            {
                "tokenized": self.tokenized,
                "data_dir": self.data_dir,
                "lemmas_path": self.lemmas_path,
                "tokenizer": self.tokenizer,
                "max_n_segments": self._max_n_segments,
            },
            path,
        )

    @classmethod
    def load_tokenized(cls, path: str) -> "RMTDocsDataset":
        state_dict = torch.load(path)
        dataset = cls(
            state_dict["data_dir"],
            state_dict["lemmas_path"],
            state_dict["tokenizer"],
            state_dict["max_n_segments"],
        )
        dataset.tokenized = state_dict["tokenized"]
        return dataset

    def __getitem__(self, index: int) -> List[Dict[str, torch.Tensor]]:
        """
        Return a list of `max_n_segments` segments with attention masks
        """
        decl_def = self.tokenized["decl_def"][index]
        proof = self.tokenized["proof"][index]
        args = self.tokenized["args"][index]

        total_length = len(decl_def) + len(proof) + sum(len(x) for x in args)

        # append random lemmas while total sequence length is less than segment_len * n_segments
        all_lemmas = list(self.lemmas.values())
        rand_lemmas = [all_lemmas[i] for i in np.random.permutation(len(self.lemmas))]

        i = 0
        sequence_length = self.segment_length * self._max_n_segments
        while total_length < sequence_length:
            if self.add_lemma_token:
                args.append(self.tokenizer('<lemma> ' + rand_lemmas[i])["input_ids"])
            else:
                args.append(self.tokenizer(rand_lemmas[i])["input_ids"])
            i += 1
            total_length += len(args[-1])

        # drop last arg if length exceeds sequence_length
        if total_length > sequence_length and args:
            args.pop()

        # shuffle args
        args = [args[i] for i in np.random.permutation(len(args))]

        input_ids_list = torch.LongTensor([*decl_def, *chain(*args), *proof])

        # if still length exceeds sequence_length just truncate last arg, but not the proof
        if len(input_ids_list) > sequence_length:
            proof = proof[:self.segment_length]
            proof_tensor = input_ids_list[-len(proof):].clone()
            input_ids_list = input_ids_list[:sequence_length].clone()
            input_ids_list[-len(proof):] = proof_tensor

        n_segments = min(self._max_n_segments, math.ceil(len(input_ids_list) / self.segment_length))
        total_length = self.segment_length * n_segments

        input_ids = torch.LongTensor(total_length).fill_(self.tokenizer.pad_token_id)

        if self.padding_side == 'right':
            input_ids[:len(input_ids_list)] = input_ids_list
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            attention_mask[len(input_ids_list):] = 0
        elif self.padding_side == 'left':
            input_ids[-len(input_ids_list):] = input_ids_list
            attention_mask = torch.zeros_like(input_ids, dtype=torch.long)
            attention_mask[-len(input_ids_list):] = 1

        input_ids = input_ids.chunk(n_segments)
        attention_mask = attention_mask.chunk(n_segments)

        output = []
        for ids, mask in zip(input_ids, attention_mask):
            labels = ids.clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100

            output.append(
                {
                    "input_ids": ids,
                    "attention_mask": mask,
                    "labels": labels,
                }
            )

        return output


def collate_docs_fn(
    batch: List[List[Dict[str, torch.Tensor]]]
) -> List[Dict[str, torch.Tensor]]:
    """
    Given a List of lists of segments return a list of batch of segments
    """
    collated = []
    for segments in zip(*batch):
        collated.append(
            {key: torch.stack([x[key] for x in segments]) for key in segments[0]}
        )

    return collated


class RMTDocsDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collate_fn = collate_docs_fn

    def __len__(self):
        return super().__len__() * self.dataset._max_n_segments

    def __iter__(self):
        for batch in super().__iter__():
            for i, segment in enumerate(batch):
                segment['batch_idx'] = i
                yield segment
