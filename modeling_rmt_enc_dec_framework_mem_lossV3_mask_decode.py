import math
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from transformers import PreTrainedModel, AutoModel

import re
import math
import copy
import types

from torch.nn import CrossEntropyLoss

class RMTEncoderDecoderForConditionalGeneration():
    def __init__(self, base_model, **rmt_kwargs):
        self.model = base_model
        self.set_params(**rmt_kwargs)


    def set_params(self, num_mem_tokens, tokenizer, **rmt_config):
        self.rmt_config = rmt_config
        self.extract_special_tokens(tokenizer)
        self.extend_word_embeddings(num_mem_tokens)
        
        self.segment_size = rmt_config['input_size'] - num_mem_tokens - tokenizer.num_special_tokens_to_add()


    def set_memory(self, memory=None):
        if memory is None:
            mem_token_ids = self.mem_token_ids.to(device=self.device)
            memory = self.embeddings(mem_token_ids)
        return memory
    
    
    def extract_special_tokens(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token = torch.tensor([tokenizer.eos_token_id])
        self.bos_token = torch.tensor([tokenizer.bos_token_id]) if 'bos_token' in tokenizer.special_tokens_map else None
    
    
    def extend_word_embeddings(self, num_mem_tokens):
        vocab_size = self.model.encoder.embed_tokens.weight.shape[0]
        extended_vocab_size = vocab_size + num_mem_tokens
        self.num_mem_tokens = num_mem_tokens
        self.mem_token_ids = torch.arange(vocab_size, vocab_size + num_mem_tokens)
        self.resize_token_embeddings(extended_vocab_size)
        self.embeddings = self.model.encoder.embed_tokens
        
        mem_start_ind = 1 if self.bos_token is not None else 0
        self.memory_position = range(mem_start_ind, mem_start_ind + num_mem_tokens)
        
        self.override_encoder_forward()

        # add reconstruction layers
        self.rec_attn = copy.deepcopy(self.model.decoder.block[-1])
        self.rec_cls = copy.deepcopy(self.model.lm_head)

        for n, p in self.rec_attn.named_parameters():
            param_name = re.sub('\.', '_', f'rec_attn_{n}')
            self.register_parameter(param_name, p)
        
        for n, p in self.rec_cls.named_parameters():
            param_name = re.sub('\.', '_', f'rec_cls_{n}')
            self.register_parameter(param_name, p)
        
    
    def override_encoder_forward(self):
        memory_forward_func, memory_layers, share_memory_layers = \
                self.rmt_config.get('memory_forward_func'), self.rmt_config.get('memory_layers'), self.rmt_config.get('share_memory_layers')

        if memory_forward_func is not None:
            new_forward = lambda *args, **kwargs: memory_forward_func(*args, **kwargs, rmt_parent=self)
            self.base_model.encoder.forward = types.MethodType(new_forward, self.base_model.encoder)

        if memory_layers is None:
            self.memory_layers = None
        else:
            if memory_layers == 'all':
                memory_layers = range(len(self.model.encoder.block))
            else:
                raise NotImplementedError
                
            if share_memory_layers:
                memory_layer = copy.deepcopy(self.model.encoder.block[0])
                self.memory_layers = [memory_layer for _ in range(len(memory_layers))]
                for n, p in memory_layer.named_parameters():
                    param_name = re.sub('\.', '_', f'memory_{n}')
                    self.register_parameter(param_name, p)
            else:
                self.memory_layers = [copy.deepcopy(self.model.encoder.block[int(l)]) for l in memory_layers]
                for ln, layer in enumerate(self.memory_layers):
                    for n, p in layer.named_parameters():
                        param_name = re.sub('\.', '_', f'{ln}_memory_{n}')
                        self.register_parameter(param_name, p)

        self.reconstruction_cls = torch.nn.Linear((self.num_mem_tokens + 1) * self.config.d_model, self.rmt_config['max_n_segments']).to(device=self.device)
        for n, p in self.reconstruction_cls.named_parameters():
            self.register_parameter(f'reconstruction_cls_{n}', p)


    def __call__(self, input_ids, **kwargs):
        memory = self.set_memory()
        memory = memory.repeat(input_ids.shape[0], 1, 1)
        segmented = self.pad_and_segment(input_ids)
        
        losses = {}
        reconstruction_loss = 0
        for seg_num, segment_input_ids in enumerate(segmented):
            if (self.rmt_config['bptt_depth'] > -1) and (len(segmented) - seg_num > self.rmt_config['bptt_depth']): 
                memory = memory.detach()

            seg_kwargs = dict(**kwargs)
            seg_kwargs['output_hidden_states'] = True
            
            non_empty_mask = [s is not None for s in segment_input_ids]
            if sum(non_empty_mask) == 0:
                continue
            input_ids = torch.stack(segment_input_ids)[non_empty_mask]
            attention_mask = self.get_attention_mask(input_ids)
            token_type_ids = self.get_token_type_ids(input_ids)
            seg_kwargs['labels'] = seg_kwargs['labels'][non_empty_mask]

            inputs_embeds = self.embeddings(input_ids)
            inputs_embeds[:, self.memory_position] = memory[non_empty_mask]
    
            seg_kwargs['inputs_embeds'] = inputs_embeds
            seg_kwargs['attention_mask'] = attention_mask
                
            out = self.model.forward(**seg_kwargs)
            memory[non_empty_mask] = out.encoder_hidden_states[-1][:, self.memory_position]

            losses[f'loss_{seg_num}'] = out['loss']

            segment_reconstruction_loss = self.segment_reconstruction_forward(input_ids, out['encoder_last_hidden_state'])
            out[f'rec_loss_{seg_num}'] = segment_reconstruction_loss
            reconstruction_loss += segment_reconstruction_loss

        # drop unnecessary hiddens to save memory
        if not kwargs.get('output_hidden_states'):
            for key in out.keys():
                if 'hidden_state' in key:
                    out[key] = None
                    
        for k, loss in losses.items():
            out[k] = loss

        if self.rmt_config['sum_loss']:
            out['loss'] = torch.stack(losses).sum(dim=0)

        out['reconstruction_loss'] = reconstruction_loss
        
        rec_coef = self.rmt_config['reconstruction_loss_coef']
        out['loss'] = out['reconstruction_loss'] * rec_coef + out['loss'] * (1 - rec_coef)

        return out
    

    def segment_reconstruction_forward(self, reconstruction_labels, encoder_out):

        attention_mask = torch.ones(encoder_out.shape[1]).to(device=self.device)
        attention_mask[self.num_mem_tokens:] = 0

        rec_attn_out = self.rec_attn(encoder_out, attention_mask=attention_mask)
        rec_logits = self.rec_cls(rec_attn_out[0])

        loss_fct = CrossEntropyLoss()
        reconstruction_loss = loss_fct(rec_logits.view(-1, rec_logits.size(-1)), reconstruction_labels.view(-1))
        reconstruction_loss

        return reconstruction_loss


    def generate(self, input_ids, **kwargs):
        memory = self.set_memory()
        memory = memory.repeat(input_ids.shape[0], 1, 1)
        segmented = self.pad_and_segment(input_ids)

        for seg_num, segment_input_ids in enumerate(segmented):                
            if (self.rmt_config['bptt_depth'] > -1) and (len(segmented) - seg_num > self.rmt_config['bptt_depth']): 
                memory = memory.detach()

            seg_kwargs = dict(**kwargs)
            seg_kwargs['output_hidden_states'] = True
            non_empty_mask = [s is not None for s in segment_input_ids]
            if sum(non_empty_mask) == 0:
                continue
            input_ids = torch.stack([s for s in segment_input_ids if s is not None])
            attention_mask = self.get_attention_mask(input_ids)
            token_type_ids = self.get_token_type_ids(input_ids)
            # seg_kwargs['labels'] = seg_kwargs['labels'][non_empty_mask]

            inputs_embeds = self.embeddings(input_ids)
            inputs_embeds[:, self.memory_position] = memory[non_empty_mask]

            seg_kwargs['inputs_embeds'] = inputs_embeds
            seg_kwargs['attention_mask'] = attention_mask

            if seg_num == len(segmented) - 1:
                out = self.model.generate(**seg_kwargs)
            else:
                for param in ['min_length', 'max_length']:
                    if param in seg_kwargs:
                        seg_kwargs.pop(param)
                        
                out = self.model.encoder(**seg_kwargs)
                memory[non_empty_mask] = out.last_hidden_state[:, self.memory_position]
        
        return out

    def pad_and_segment(self, input_ids, **kwargs):       
        segmented_batch = []
        for seq in input_ids:
            seq = seq[(seq != self.pad_token_id) & (seq != self.eos_token.item())]
            if self.bos_token is not None:
                seq = seq[seq != self.bos_token_id]
            seq = seq[:self.segment_size * self.rmt_config['max_n_segments']]

            n_seg = math.ceil(len(seq) / self.segment_size)
            input_segments = torch.chunk(seq, n_seg)
            input_segments = [self.pad_add_special_tokens(t, self.rmt_config['input_size']) for t in input_segments]

            segmented_batch.append(input_segments)
    
        # batch of segments -> segmented batch 
        # + align segments to right border
        # so that the last segment is always non-empty
        segmented_batch = [[s[::-1][i] if len(s) > i else None for s in segmented_batch] \
                            for i in range(self.rmt_config['max_n_segments'])][::-1]
        return segmented_batch
    
    
    def pad_add_special_tokens(self, tensor, segment_size):
        input_elements = []
        if self.bos_token is not None:
            input_elements.append(self.bos_token.to(device=self.device))

        input_elements += [
                        self.mem_token_ids.to(device=self.device),
                        tensor.to(device=self.device),
                        self.eos_token.to(device=self.device)
                        ]
        tensor = torch.cat(input_elements)
        
        pad_size = segment_size - tensor.shape[0]
        if pad_size > 0:
            tensor = F.pad(tensor, (0, pad_size))                  
        return tensor
    
    
    def get_attention_mask(self, tensor):
        mask = torch.ones_like(tensor)
        mask[tensor == self.pad_token_id] = 0
        return mask
        
    
    def get_token_type_ids(self, tensor):
        return torch.zeros_like(tensor)


    def to(self, device):
        self.model = self.model.to(device)
        
    
    def cuda(self):
        self.model.cuda()


    def __getattr__(self, attribute):
        return getattr(self.model, attribute)