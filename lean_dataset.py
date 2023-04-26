import math
from pathlib import Path
from typing import Dict, List

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
        self.data_dir = data_dir
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

    def split_to_segments(self, segment_len: int) -> None:
        self._segment_len = segment_len
        self.input_ids = []
        self.attn_masks = []
        self._sequence_id = []
        for i, tensor in enumerate(self.tokenized_documents):

            n_segments = math.ceil(len(tensor) / segment_len)

            if n_segments == 0:
                continue

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

            if not self.drop_last:
                indices = torch.arange(total_segments, total_segments + n_segments)
            elif n_segments >= self._max_n_segments:
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

        if not self.drop_last:
            # drop segments with different segment_id and pad
            i = 0
            segment_ids = self._sequence_id[idx : idx + self._max_n_segments]
            while i < len(segment_ids) and segment_ids[i] == segment_ids[i]:
                i += 1

            pad_len = self._max_n_segments - i
            if pad_len:
                input_ids = input_ids[:i]
                attn_mask = attn_mask[:i]

                input_ids.extend(
                    [
                        torch.LongTensor(self._segment_len).fill_(
                            self.tokenizer.pad_token_id
                        )
                        for _ in range(pad_len)
                    ]
                )
                attn_mask.extend(
                    [
                        torch.zeros(self._segment_len, dtype=torch.long)
                        for _ in range(pad_len)
                    ]
                )

        return [
            {"input_ids": ids, "attention_mask": mask}
            for ids, mask in zip(input_ids, attn_mask)
        ]


def collate_docs_fn(
    batch: List[List[Dict[str, torch.Tensor]]]
) -> List[Dict[str, torch.Tensor]]:
    """
    Given a List of lists of segments return a list of batch of segments
    """
    collated = []
    for segments in zip(*batch):
        collated.append(
            {
                "input_ids": torch.stack([x["input_ids"] for x in segments]),
                "attention_mask": torch.stack([x["attention_mask"] for x in segments]),
            }
        )

    return collated


class RMTDocsDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        assert kwargs.get('collate_fn', None) is None
        super().__init__(*args, **kwargs)
        self.collate_fn = collate_docs_fn

    def __len__(self):
        return super().__len__() * self.dataset._max_n_segments

    def __iter__(self):
        for batch in super().__iter__():
            for segment in batch:
                yield segment

