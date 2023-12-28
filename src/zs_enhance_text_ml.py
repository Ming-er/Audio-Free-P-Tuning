from CLAPWrapper import CLAPWrapper
import torch
import numpy as np
from copy import deepcopy
import torch.nn as nn
from fsd19_dataset import *
from dcase19_dataset import *
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from importlib_resources import files
from models.utils import read_config_as_args
from sklearn.metrics import average_precision_score
from torch.utils.data import Dataset, DataLoader
import torch

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def cal_map(y_preds, y_labels):
    AP = []
    for i in range(y_labels.shape[1]):
        AP.append(average_precision_score(y_labels[:, i], y_preds[:, i]))
    return np.mean(AP)

# equation 3 in the paper
def ranking_loss(y_pred, y_true, scale_ = 5.0, margin_ = 1):
    y_pred *= scale_
    y_true_ = y_true.float()
    tmp = margin_ - y_pred[:, None, :] + y_pred[:, :, None]
    partial_losses = torch.maximum(torch.zeros_like(tmp), tmp)
    loss = partial_losses * y_true_[:, None, :] * (1 - y_true_[:, :, None])
    loss = torch.sum(loss, dim=-1)
    loss = torch.sum(loss, dim=-1)
    return torch.mean(loss)

def eval_one_epoch(eval_dataloader, wrapper, device, y):
    wrapper.clap.eval()
    wrapper.prompts_embeddings.eval()
    wrapper.prompts_embeddings_frame.eval()

    label_embeds, _ = wrapper.get_text_embeddings(y) 
    label_embeds_frame, _ = wrapper.get_text_embeddings(y, frame=True)  
      
    y_preds, y_preds_frame, y_preds_ensemble, y_labels = [], [], [], []
    for _, (audio, multi_hot_targets) in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
        audio = audio.to(device, non_blocking=True)

        with torch.no_grad():    
            logits = wrapper.compute_at_similarity(audio, label_embeds, is_spec_augment=False, ml=True)
            logits_frame = wrapper.aggre_frame_sim(audio, label_embeds_frame, is_spec_augment=False, ml=True)

            logits_ensemble = logits + logits_frame
            y_pred = logits.detach().cpu().numpy()
            y_pred_frame = logits_frame.detach().cpu().numpy()
            y_pred_ensemble = logits_ensemble.detach().cpu().numpy()

        y_preds.append(y_pred)
        y_preds_frame.append(y_pred_frame)
        y_preds_ensemble.append(y_pred_ensemble)
        y_labels.append(multi_hot_targets.detach().cpu().numpy())

    y_labels, y_preds, y_preds_frame, y_preds_ensemble = np.concatenate(y_labels, axis=0),\
                                                         np.concatenate(y_preds, axis=0), np.concatenate(y_preds_frame, axis=0), np.concatenate(y_preds_ensemble, axis=0)
    mAP = cal_map(y_preds, y_labels)
    print('mAP coarse-grained: {}'.format(mAP))
    mAP = cal_map(y_preds_frame, y_labels)
    print('mAP fine-grained: {}'.format(mAP))
    mAP = cal_map(y_preds_ensemble, y_labels)
    print('mAP ensemble: {}'.format(mAP))
    return mAP

def train_one_epoch(train_dataloader, wrapper, optimizer, device, y):

    wrapper.clap.train()
    wrapper.frozen_caption_encoder.eval()
    wrapper.prompts_embeddings.train()
    wrapper.prompts_embeddings_frame.train()

    for _, (text, multi_hot_targets) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        multi_hot_targets = multi_hot_targets.to(device)

        label_embeds, _ = wrapper.get_text_embeddings(y)
        label_embeds_frame, _ = wrapper.get_text_embeddings(y, frame=True)

        logits = wrapper.compute_tt_similarity(text, label_embeds, ml=True)
        logits_frame = wrapper.compute_tt_similarity_frame(text, label_embeds_frame, ml=True)

        loss_st = ranking_loss(logits, multi_hot_targets)
        loss_st_frame = ranking_loss(logits_frame, multi_hot_targets)

        optimizer.zero_grad()
        loss = loss_st + loss_st_frame
        loss.backward(create_graph=False)     
        optimizer.step()

    return

config_as_str = files('configs').joinpath('config.yml').read_text()
args = read_config_as_args(config_as_str, is_config_str=True)
device = "cuda" if args.use_cuda else "cpu"

wrapper = CLAPWrapper(args)

if args.dataset == 'fsd':
    train_dataset = FSD19_Text(args.fsd_text_path)
    eval_dataset = FSD19_eval(root=args.fsd_audio_path, download=False)
else:
    train_dataset = DCASE19_Text(args.dcase_text_path)
    eval_dataset = DCASE19_val(root=args.dcase_audio_path, download=False)

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
num_class = eval_dataset.class_num

prompt = ' {}'
y = [prompt.format(x) for x in eval_dataset.classes]

optimizer = torch.optim.AdamW([{'params': wrapper.prompts_embeddings.parameters()}, {'params': wrapper.prompts_embeddings_frame.parameters()}], lr=args.lr, betas=(0.9, 0.98), eps=1e-06)

best_map = 0
for e in range(args.epochs):
    print("Train Epoch {}: ".format(e))
    train_one_epoch(train_dataloader, wrapper, optimizer, device, y)
    print("Infer Epoch {}: ".format(e))
    mAP = eval_one_epoch(eval_dataloader, wrapper, device, y)
    if mAP > best_map:
        best_map = mAP
        torch.save(wrapper.prompts_embeddings.state_dict(), 'best_model_coarse.pt')
        torch.save(wrapper.prompts_embeddings_frame.state_dict(), 'best_model_fine.pt')

