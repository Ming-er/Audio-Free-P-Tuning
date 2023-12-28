import random
import torchaudio
from torch._six import string_classes
import collections
import re
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
from models.clap import CLAP
import math
import torchaudio.transforms as T
import os
import torch
from copy import deepcopy


class CLAPWrapper():
    def __init__(self, args):
        self.model_fp = args.model_fp
        self.use_cuda = args.use_cuda
        self.args = args
        self.clap, self.tokenizer = self.load_clap() 
        self.frozen_caption_encoder = deepcopy(self.clap.caption_encoder)
        self.freeze_weight(self.clap)
        self.freeze_weight(self.frozen_caption_encoder)
        self.prompts_token, self.prompts_embeddings = self.init_prompt(args.prompt_len, args.transformer_embed_dim) if args.prompt else None
        self.prompts_token_frame, self.prompts_embeddings_frame = self.init_prompt(args.prompt_len, args.transformer_embed_dim) if args.prompt else None

    def load_clap(self):
        if 'bert' in self.args.text_model:
            self.token_keys = ['input_ids', 'token_type_ids', 'attention_mask']
        else:
            self.token_keys = ['input_ids', 'attention_mask']

        clap = CLAP(
            audioenc_name=self.args.audioenc_name,
            sample_rate=self.args.sampling_rate,
            window_size=self.args.window_size,
            hop_size=self.args.hop_size,
            mel_bins=self.args.mel_bins,
            fmin=self.args.fmin,
            fmax=self.args.fmax,
            classes_num=self.args.num_classes,
            out_emb=self.args.out_emb,
            text_model=self.args.text_model,
            transformer_embed_dim=self.args.transformer_embed_dim,
            d_proj=self.args.d_proj
        )

        model_state_dict = torch.load(self.model_fp, map_location=torch.device('cpu'))['model']
        clap.load_state_dict(model_state_dict, strict=False)

        tokenizer = AutoTokenizer.from_pretrained(self.args.text_model)

        if self.use_cuda and torch.cuda.is_available():
            clap = clap.cuda()
            
        return clap, tokenizer

    def init_prompt(self, num_tokens, transformer_embed_dim):
        prefix_tokens = torch.arange(num_tokens).long()
        prefix_encoder = torch.nn.Embedding(num_tokens, transformer_embed_dim)
        torch.nn.init.normal_(prefix_encoder.weight, std=0.02)
        if self.use_cuda and torch.cuda.is_available():
            prefix_tokens = prefix_tokens.cuda()
            prefix_encoder = prefix_encoder.cuda()
        return prefix_tokens, prefix_encoder

    def freeze_weight(self, model):
        for name, param in model.named_parameters():
            param.requires_grad = False
        
    def norm_embeds(self, vec):
        return F.normalize(vec, dim=-1)
        
    def get_text_embeddings(self, text, frame=False):
        text_input = self.tokenizer(text=text, add_special_tokens=True, max_length=self.args.text_len - self.args.prompt_len, padding='max_length', return_tensors="pt")
        if not frame:
            for item in text_input.keys():
                if self.use_cuda and torch.cuda.is_available():
                    text_input[item] = text_input[item].cuda()

            prompts = self.prompts_embeddings(self.prompts_token)
            prompts = prompts.unsqueeze(0).expand(len(text), -1, -1)
            prompt_attention_mask = torch.ones(len(text), self.args.prompt_len).cuda() if self.use_cuda and torch.cuda.is_available() else torch.ones(len(text), self.args.prompt_len)
            text_embeds = self.clap.encode_text(text_input, use_prompt=self.args.prompt, prompts=prompts, prompt_attention_mask=prompt_attention_mask)
            return text_embeds, prompts
        else:
            for item in text_input.keys():
                if self.use_cuda and torch.cuda.is_available():
                    text_input[item] = text_input[item].cuda()

            prompts = self.prompts_embeddings_frame(self.prompts_token)
            prompts = prompts.unsqueeze(0).expand(len(text), -1, -1)
            prompt_attention_mask = torch.ones(len(text), self.args.prompt_len).cuda() if self.use_cuda and torch.cuda.is_available() else torch.ones(len(text), self.args.prompt_len)
            text_embeds = self.clap.encode_text(text_input, use_prompt=self.args.prompt, prompts=prompts, prompt_attention_mask=prompt_attention_mask)
            return text_embeds, prompts
    
    def get_audio_embeddings(self, audio, is_spec_augment=False, mixup_lambda=None, mixup_perm=None):
        audio_embeds = self.clap.encode_audio(audio, is_spec_augment=is_spec_augment, mixup_lambda=mixup_lambda, mixup_perm=mixup_perm)
        return audio_embeds

    def compute_at_similarity(self, audio, label_embeds, is_spec_augment=False, ml=False):
        audio_embeds = self.clap.encode_audio(audio, is_spec_augment=is_spec_augment)
        temp_clip = self.clap.return_temp()
        norm_audio_embeds = self.norm_embeds(audio_embeds)
        norm_label_embeds = self.norm_embeds(label_embeds)
        score = norm_audio_embeds @ norm_label_embeds.t()
        if not ml:
            return score * temp_clip
        else:
            return score
        
    def aggre_frame_sim(self, audio, label_embeds, is_spec_augment=False, T=0.10, ml=False):
        frame_embeds = self.clap.encode_frame(audio, is_spec_augment=is_spec_augment)
        temp_clip = self.clap.return_temp()
        norm_frame_embeds = self.norm_embeds(frame_embeds)
        norm_label_embeds = self.norm_embeds(label_embeds).squeeze(0)
        frame_score = (norm_frame_embeds @ norm_label_embeds.t()) # (bs, T, nclass)
        aggregate_attn = F.softmax(frame_score / T, dim=1)
        score = (frame_score * aggregate_attn).sum(dim=1)
        if not ml:
            return score * temp_clip
        else:
            return score
    
    def compute_tt_similarity(self, text, label_embeds, is_spec_augment=False, ml=False):
        text_input = self.tokenizer(text=text, add_special_tokens=True, max_length=self.args.text_len, padding='max_length', return_tensors="pt")
        for item in text_input.keys():
            if self.use_cuda and torch.cuda.is_available():
                text_input[item] = text_input[item].cuda()
        text_embeds = self.frozen_caption_encoder(text_input)
        temp_clip = self.clap.return_temp()
        norm_text_embeds = self.norm_embeds(text_embeds)
        norm_label_embeds = self.norm_embeds(label_embeds)
        score = norm_text_embeds @ norm_label_embeds.t()
        if not ml:
            return score * temp_clip
        else:
            return score
        
    def compute_tt_similarity_frame(self, text, label_embeds, is_spec_augment=False, ml=False, T=0.10):
        text_input = self.tokenizer(text=text, add_special_tokens=True, max_length=self.args.text_len, padding='max_length', return_tensors="pt")
        for item in text_input.keys():
            if self.use_cuda and torch.cuda.is_available():
                text_input[item] = text_input[item].cuda()
        word_embeds = self.frozen_caption_encoder(text_input, return_word=True)
        temp_clip = self.clap.return_temp()
        norm_words_embeds = self.norm_embeds(word_embeds)
        norm_label_embeds = self.norm_embeds(label_embeds).squeeze(0)
        word_score = (norm_words_embeds @ norm_label_embeds.t())
        attn_mask = text_input["attention_mask"][:, 1:].unsqueeze(-1)
        aggregate_attn = F.softmax(word_score * attn_mask / T, dim=1)
        score = (word_score * aggregate_attn).sum(dim=1)
        if not ml:
            return score * temp_clip
        else:
            return score
            
    def get_original_text_embedding(self, text):
        text_input = self.tokenizer(text=text, add_special_tokens=True, max_length=self.args.text_len, padding='max_length', return_tensors="pt")
        for item in text_input.keys():
            if self.use_cuda and torch.cuda.is_available():
                text_input[item] = text_input[item].cuda()
        text_embeds = self.frozen_caption_encoder(text_input)
        return text_embeds

    def return_temp(self):
        return self.clap.return_temp()