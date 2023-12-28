import numpy as np
import torch
import math
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel
from .audio import get_audio_encoder

class Projection(nn.Module):
    def __init__(self, d_in: int, d_out: int, p: float=0.5) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds

class AudioEncoder(nn.Module):
    def __init__(self, audioenc_name:str, d_in: int, d_out: int, sample_rate: int, window_size: int,
            hop_size: int, mel_bins: int, fmin: int, fmax: int, classes_num: int) -> None:
        super().__init__()

        audio_encoder = get_audio_encoder(audioenc_name)

        self.base = audio_encoder(
            sample_rate, window_size,
            hop_size, mel_bins, fmin, fmax,
            classes_num, d_in)

        self.projection = Projection(d_in, d_out)

    def forward(self, x, is_spec_augment=False, mixup_lambda=None, mixup_perm=None):
        out_dict = self.base(x, is_spec_augment=is_spec_augment, mixup_lambda=mixup_lambda, mixup_perm=mixup_perm)
        audio_features, audio_classification_output = out_dict['embedding'], out_dict['clipwise_output']
        projected_vec = self.projection(audio_features)
        return projected_vec, audio_classification_output

class TextEncoder(nn.Module):
    def __init__(self, d_out: int, text_model: str, transformer_embed_dim: int) -> None:
        super().__init__()
        self.base = AutoModel.from_pretrained(text_model) 
        self.projection = Projection(transformer_embed_dim, d_out)

    def forward(self, x, use_prompt=False, prompts=None, prompt_attention_mask=None):
        if not use_prompt:
            out = self.base(**x)[0]
            out = out[:, 0, :]  # get CLS token output
            projected_vec = self.projection(out)
            return projected_vec
        else:           
            raw_embedding = self.base.embeddings(input_ids=x.input_ids, token_type_ids=x.token_type_ids)
            inputs_embeds = torch.cat([raw_embedding[:, 0, :].unsqueeze(1), prompts, raw_embedding[:, 1:, :]], dim=1)
            attention_mask = torch.cat([prompt_attention_mask, x.attention_mask], dim=1)
            out  = self.base(attention_mask=attention_mask, inputs_embeds=inputs_embeds)[0]
            out = out[:, 0, :]  # get CLS token output
            projected_vec = self.projection(out)
            return projected_vec

class CLAP(nn.Module):
    def __init__(self,
                # audio
                audioenc_name: str,
                sample_rate: int, 
                window_size: int, 
                hop_size: int, 
                mel_bins: int, 
                fmin: int, 
                fmax: int, 
                classes_num: int, 
                out_emb: int,
                # text
                text_model: str,
                transformer_embed_dim: int,
                # common
                d_proj: int,
                ):
        super().__init__()

        
        self.audio_encoder = AudioEncoder(
            audioenc_name, out_emb, d_proj,
            sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num)

        self.caption_encoder = TextEncoder(
            d_proj, text_model, transformer_embed_dim
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_audio(self, audio, is_spec_augment=False, mixup_lambda=None, mixup_perm=None):
        audio_embed, _ = self.audio_encoder(audio, is_spec_augment=is_spec_augment, mixup_lambda=mixup_lambda, mixup_perm=mixup_perm)
        return audio_embed

    def encode_text(self, text, use_prompt=False, prompts=None, prompt_attention_mask=None):
        caption_embed = self.caption_encoder(text, use_prompt=use_prompt, prompts=prompts, prompt_attention_mask=prompt_attention_mask)
        return caption_embed
    
    def return_temp(self):
        with torch.no_grad():
            self.logit_scale.clamp_(0, math.log(100))
        return self.logit_scale.exp()
    
    def forward(self, audio, text, is_spec_augment=False):
        audio_embed, _ = self.audio_encoder(audio, is_spec_augment=is_spec_augment)
        caption_embed = self.caption_encoder(text)

        return caption_embed, audio_embed, self.logit_scale.exp()