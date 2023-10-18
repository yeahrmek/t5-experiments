import json
import math
import random
import copy
import joblib
from itertools import chain
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from datasets import Dataset, set_caching_enabled
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
from tqdm import tqdm, trange

import re
import friendlywords

import time

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
        short_proofs_only = False,
        every_segment_def = True,
        exclude_relevant_lemmas = False,
        use_random_lemmas_names = False,
        random_names_choices = 100 # how many random choices of lemmas names to use (we can not compute this each time)
    ):
        assert padding_side in ['left', 'right']
        if padding_side == 'left':
            print('for counting loss on definition and the rest separately we only support right padding')
            raise NotImplementedError
        self.data_dir = str(Path(data_dir).resolve())
        self.lemmas_path = str(Path(lemmas_path).resolve())
        self.tokenizer = tokenizer
        self._max_n_segments = max_n_segments
        self.segment_length = segment_length
        self.padding_side = padding_side
        self.short_proofs_only = short_proofs_only
        self.every_segment_def = every_segment_def
        self.exclude_relevant_lemmas = exclude_relevant_lemmas
        self.use_random_lemmas_names = use_random_lemmas_names
        self.random_names_choices = random_names_choices
        
#         self.bos = self.tokenizer.vocab['[BOS]']
#         self.eos = self.tokenizer.vocab['[EOS]']

        self.dataset = Dataset.from_parquet(
            [str(x) for x in Path(data_dir).glob("*.parquet")]
        )
        
        #if self.short_proofs_only:
        #    self.dataset = self.dataset.filter(
        #        lambda x: len(self.tokenizer.encode(f"[PROOFSTEP] {x['full_proof']}", 
        #                    truncation=False,
        #                    padding=False)) <= self.segment_length
        #    )
        

        with open(lemmas_path) as fh:
            lemmas = json.load(fh)
            self.lemmas = {x.split('.')[-1]: lemmas[x] for x in lemmas.keys()}
            
        self.tokenized = {}
        self.lemmas_tokenized = []            

    def __len__(self) -> int:
        if self.use_random_lemmas_names:
            return len(self.tokenized["decl_def"][0])
        return len(self.tokenized.get("decl_def", []))

    def set_max_n_segments(self, max_n_segments: int) -> None:
        self._max_n_segments = max_n_segments
        

#     def _filter_shorts_fun(self, x):
#         proof_len = len(self.tokenizer.encode(f"[PROOFSTEP] {x['full_proof']}", 
#                         truncation=False,
#                         padding=False))
#         def_len = len(self.tokenizer.encode(f"[GOAL] {x['decl_def']}", 
#                         truncation=False,
#                         padding=False))
#         ids = self.tokenizer.encode(f"[PROOFSTEP] {x['full_proof']}", 
#                         truncation=False,
#                         padding=False)
#         if True:
#             x = 1/0
#             print()
#         return (proof_len + def_len <= self.segment_length - 2) and (def_len <= self.segment_length // 2)

    def _filter_shorts_fun(self, x):
        proof_len = len(self.tokenizer.encode(f"[PROOFSTEP] {x['full_proof']}", 
                        truncation=False,
                        padding=False))
        def_len = len(self.tokenizer.encode(f"[PROOFSTEP] {x['decl_def']}", 
                        truncation=False,
                        padding=False))
        return (proof_len + def_len <= self.segment_length) and (def_len <= self.segment_length // 2)
    
    def filter_shorts(self) -> None:
        """
        Drop long proofs and definitions
        """

        def _filter_shorts_fun(x):
            proof_len = len(self.tokenizer.encode(f"[PROOFSTEP] {x['full_proof']}", 
                            truncation=False,
                            padding=False))
            def_len = len(self.tokenizer.encode(f"[GOAL] {x['decl_def']}", 
                            truncation=False,
                            padding=False))
            ids = self.tokenizer.encode(f"[PROOFSTEP] {x['full_proof']}", 
                            truncation=False,
                            padding=False)
            return (proof_len + def_len <= self.segment_length - 2) and (def_len <= self.segment_length // 2)
        self.dataset = self.dataset.filter(_filter_shorts_fun, load_from_cache_file=False)
    
    def _tokenize(self, decl_def, proof, args_list, lemmas):
        decl_def_tokenized = self.tokenizer(
            [f"[GOAL] {x}" for x in decl_def],
            truncation=False,
            padding=False,
        )["input_ids"]

        proof_tokenized = self.tokenizer(
            [f"[PROOFSTEP] {x}" for x in proof],
            truncation=False,
            padding=False,
        )["input_ids"]

        args_defs_tokenized = []
        for args in args_list:
            args_defs = [
                f"[AUX_LEMMA] {x.strip()}" for x in args.split("<lemma>") if x.strip()
            ]
            if args_defs:
                args_defs_tokenized.append(
                    self.tokenizer(args_defs, truncation=False, padding=False)[
                        "input_ids"
                    ]
                )
            else:
                args_defs_tokenized.append([])
        
        lemmas_tokenized = []
        for lemma in lemmas.values():
            tokenized = self.tokenizer.encode('[AUX_LEMMA] ' + lemma)
            if len(tokenized) <= self.segment_length:
                lemmas_tokenized.append(tokenized)
        
        return decl_def_tokenized, proof_tokenized, args_defs_tokenized, lemmas_tokenized
    
    def tokenize(self) -> None:
        if self.use_random_lemmas_names:
            self.tokenized["decl_def"] = []
            self.tokenized["proof"] = []
            self.tokenized["args"] = []
            self.lemmas_tokenized = []
            
            _dataset = copy.deepcopy(self.dataset)
            for i in trange(self.random_names_choices):
                rand_names = {
                    x: '_'.join([y[:4] for y in friendlywords.generate(3, as_list=True)])[:len(x)] for x in self.lemmas.keys()
                }
#                 def repl_fun(matchobj):
#                     name = matchobj.group()
#                     if len(name) > 4 and bool(re.fullmatch(r"[a-zA-Z_]+", name)) and name in rand_names:
#                         return rand_names[name]
#                     return name
                
#                 def randomize_dataset(data):
#                     data["decl_def"] = re.sub(r'\b[a-zA-Z_]+\b', repl_fun, data["decl_def"])
#                     data["full_proof"] = re.sub(r'\b[a-zA-Z_]+\b', repl_fun, data["full_proof"])
#                     data["args_defs"] = re.sub(r'\b[a-zA-Z_]+\b', repl_fun, data["args_defs"])
#                     return data
#                 dataset = copy.deepcopy(self.dataset)
#                 dataset = dataset.map(randomize_dataset)
#                 #dataset = self._filter_shorts(dataset)
#                 randomized_lemmas = {rand_names[x]: re.sub(r'\b[a-zA-Z_]+\b', repl_fun, self.lemmas[x]) for x in self.lemmas.keys()}

                # changing only names
                def repl_fun_for_lemmas(matchobj):
                    t = matchobj.group(1) # lemma, theorem or def
                    name = matchobj.group(2)
                    if len(name) > 4 and bool(re.fullmatch(r"[a-zA-Z_]+", name)) and name in rand_names:
                        return f"{t} {rand_names[name]}"
                    return f"{t} {name}"
                
                randomized_lemmas = {x: re.sub(r'^(def|lemma|theorem) ([^\s]+)', repl_fun_for_lemmas, self.lemmas[x]) for x in self.lemmas.keys()}
                def repl_fun(matchobj):
                    t = matchobj.group(1) # lemma, theorem or def
                    prefix = matchobj.group(2)
                    name = matchobj.group(3)
                    if len(name) > 4 and name in rand_names:
                        name = rand_names[name]
                    if not prefix:
                        prefix = ''
                    return f"<lemma> {t} {prefix}{name}"
            
                def randomize_dataset(data):
                    args_names = [x.strip().split()[1] for x in data["args_defs"].split('<lemma>') if x.strip()]
                    data["args_defs"] = re.sub(r'<lemma> ([a-z]+) ([a-zA-Z0-9_\.]+\.)?([a-zA-Z0-9_]+\b)', repl_fun, data["args_defs"])
                    def repl_fun_only_relevant(matchobj):
                        name = matchobj.group()
                        if len(name) > 4 and bool(re.fullmatch(r"[a-zA-Z0-9_]+", name)) and name in args_names:
                            return rand_names[name]
                        return name
                    data["decl_def"] = re.sub(r'\b[a-zA-Z0-9_]+\b', repl_fun_only_relevant, data["decl_def"])
                    data["full_proof"] = re.sub(r'\b[a-zA-Z0-9_]+\b', repl_fun_only_relevant, data["full_proof"])
                    return data
                
                dataset = copy.deepcopy(_dataset)
                dataset = dataset.map(randomize_dataset, load_from_cache_file=False)
                def _filter_shorts_fun(x, tokenizer, segment_length): # it's impossible to avoid duplication here
                    proof_len = len(tokenizer.encode(f"[PROOFSTEP] {x['full_proof']}", 
                                    truncation=False,
                                    padding=False))
                    def_len = len(tokenizer.encode(f"[PROOFSTEP] {x['decl_def']}", 
                                    truncation=False,
                                    padding=False))
                    return (proof_len + def_len <= segment_length) and (def_len <= segment_length // 2)
                
                dataset = dataset.filter(_filter_shorts_fun, fn_kwargs={'tokenizer': self.tokenizer, 'segment_length': self.segment_length}, load_from_cache_file=False)
                
                decl_def_tokenized, proof_tokenized, args_defs_tokenized, lemmas_tokenized = self._tokenize(
                    dataset["decl_def"], dataset["full_proof"], dataset["args_defs"], randomized_lemmas
                )
                self.tokenized["decl_def"].append(decl_def_tokenized)
                self.tokenized["proof"].append(proof_tokenized)
                self.tokenized["args"].append(args_defs_tokenized)
                self.lemmas_tokenized.append(lemmas_tokenized)           
        else:
            self.tokenized["decl_def"], self.tokenized["proof"], self.tokenized["args"], self.lemmas_tokenized = self._tokenize(
                self.dataset["decl_def"], self.dataset["full_proof"], self.dataset["args_defs"], self.lemmas
            )

    def save_tokenized(self, path: str) -> None:
        torch.save(
            {
                "tokenized": self.tokenized,
                "data_dir": self.data_dir,
                "lemmas_path": self.lemmas_path,
                "tokenizer": self.tokenizer,
                "max_n_segments": self._max_n_segments,
                "lemmas_tokenized": self.lemmas_tokenized
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
        dataset.lemmas_tokenized = state_dict["lemmas_tokenized"]
        return dataset
    
    def _pack_agrs_with_irrelevant(self, args, lemmas_tokenized, n_segments, capacity):
        args = args.copy()
        random.shuffle(args)
        bins = []
        bins_sizes = []
        for i in range(n_segments):
            bins.append([])
            bins_sizes.append(0)
            
        if args and not self.exclude_relevant_lemmas:
            for arg in args:
                for i in np.random.permutation(n_segments):
                    if bins_sizes[i] + len(arg) <= capacity:
                        bins[i].append(arg)
                        bins_sizes[i] += len(arg)
                        break
                #else:
                #    raise PackingError
        
        while True:
            arg = random.choice(lemmas_tokenized)
            for i in np.random.permutation(n_segments):
                if bins_sizes[i] + len(arg) <= capacity:
                    bins[i].append(arg)
                    bins_sizes[i] += len(arg)
                    break
            else:
                break
        
        for i in range(n_segments):
            random.shuffle(bins[i])
            bins[i] = [token for arg in bins[i] for token in arg]
        
        return bins
        
    
    def __getitem__(self, index: int) -> List[Dict[str, torch.Tensor]]:
        """
        Return a list of `max_n_segments` segments with attention masks
        """
        #print('in getitem', len(self.tokenized['decl_def']))
        #print(self.tokenized["args"])
        #print(len(self.tokenized["args"]))
        #print(len(self.tokenized["args"][0]))
        #print(len(self.tokenized["args"][0][0]))
        if self.use_random_lemmas_names:
            idx = random.randint(0, self.random_names_choices - 1)
            index %= len(self.tokenized["decl_def"][idx]) # some randomized datasets contain less rows than original one
            decl_def = self.tokenized["decl_def"][idx][index].copy()
            proof = self.tokenized["proof"][idx][index].copy()
            args = self.tokenized["args"][idx][index].copy()
            lemmas_tokenized = self.lemmas_tokenized[idx].copy()
        else:
            decl_def = self.tokenized["decl_def"][index].copy()
            proof = self.tokenized["proof"][index].copy()
            args = self.tokenized["args"][index].copy()
            lemmas_tokenized = self.lemmas_tokenized.copy()
        
        

        # pack lemmas into segments
        packing = self._pack_agrs_with_irrelevant(args, lemmas_tokenized, self._max_n_segments - 1, self.segment_length - len(decl_def))
        #packing = [[] for i in range(self._max_n_segments - 1)]
        
        packing.append(proof) # proof goes in separate segment
        
        output = []
        for args_proof_ids in packing:
            ids = torch.LongTensor(self.segment_length).fill_(self.tokenizer.pad_token_id)
#             ids[0] = self.bos
#             ids[1:1 + len(decl_def)] = torch.LongTensor(decl_def)
#             ids[1 + len(decl_def):1 + len(decl_def) + len(args_proof_ids)] = torch.LongTensor(args_proof_ids)
#             ids[1 + len(decl_def) + len(args_proof_ids)] = self.eos

            ids[:len(decl_def)] = torch.LongTensor(decl_def)
            ids[len(decl_def):len(decl_def) + len(args_proof_ids)] = torch.LongTensor(args_proof_ids)
            
            attention_mask = torch.ones_like(ids, dtype=torch.long)
            #attention_mask[len(decl_def) + len(args_proof_ids) + 2:] = 0
            attention_mask[len(decl_def) + len(args_proof_ids):] = 0
            
            ids[:len(decl_def)] = torch.LongTensor(decl_def)
            ids[len(decl_def):len(decl_def) + len(args_proof_ids)] = torch.LongTensor(args_proof_ids)
            attention_mask = torch.ones_like(ids, dtype=torch.long)
            attention_mask[len(decl_def) + len(args_proof_ids):] = 0
            labels = ids.clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100

            output.append(
                {
                    "input_ids": ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "def_len": torch.tensor(len(decl_def))
                }
            )
        
        #output = [copy.deepcopy(output[0])] + output
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


class RMTDocsAllAtOnceDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collate_fn = collate_docs_fn
