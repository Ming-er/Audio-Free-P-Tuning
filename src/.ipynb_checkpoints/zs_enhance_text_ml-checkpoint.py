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
import math
from torch.optim.lr_scheduler import LambdaLR
import torch
from torch.optim.lr_scheduler import _LRScheduler

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class WarmupCosineSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, t_total, cycles=0.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch, verbose=True
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        return 1

def cal_map(y_preds, y_labels):
    AP = []
    for i in range(y_labels.shape[1]):
        AP.append(average_precision_score(y_labels[:, i], y_preds[:, i]))
    return np.mean(AP)

config_as_str = files('configs').joinpath('config.yml').read_text()
args = read_config_as_args(config_as_str, is_config_str=True)
device = "cuda" if args.use_cuda else "cpu"

def ranking_loss(y_pred, y_true, scale_ = 5.0, margin_ = 1):
    y_pred *= scale_
    y_true_ = y_true.float()
    tmp = margin_ - y_pred[:, None, :] + y_pred[:, :, None]
    partial_losses = torch.maximum(torch.zeros_like(tmp), tmp)
    loss = partial_losses * y_true_[:, None, :] * (1 - y_true_[:, :, None])
    loss = torch.sum(loss, dim=-1)
    loss = torch.sum(loss, dim=-1)
    return torch.mean(loss)

def eval_one_epoch(dataset, wrapper, y):
    eval_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    wrapper.clap.eval()
    wrapper.prompts_embeddings.eval()
    wrapper.prompts_embeddings_frame.eval()
    label_embeds, _ = wrapper.get_text_embeddings(y) 
    label_embeds_frame, _ = wrapper.get_text_embeddings(y, frame=True)    
    y_preds, y_preds_frame, y_preds_ensemble, y_labels = [], [], [], []
    for i, (audio, multi_hot_targets) in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
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
    print('mAP {}'.format(mAP))
    mAP = cal_map(y_preds_frame, y_labels)
    print('mAP Frame {}'.format(mAP))
    mAP = cal_map(y_preds_ensemble, y_labels)
    print('mAP Ensemble {}'.format(mAP))
    return mAP

def train_one_epoch(args, train_dataloader, wrapper, epoch, optimizer, lr_scheduler, device, num_class, y):
    wrapper.clap.train()
    wrapper.frozen_caption_encoder.eval()
    wrapper.prompts_embeddings.train()
    wrapper.prompts_embeddings_frame.train()
    for i, (text, multi_hot_targets) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        multi_hot_targets = multi_hot_targets.to(device)
        label_embeds, _ = wrapper.get_text_embeddings(y)
        label_embeds_frame, _ = wrapper.get_text_embeddings(y, frame=True)
        # norm_ori_label_embeds = wrapper.norm_embeds(ori_label_embeds)
        # loss_func = nn.BCELoss()
        logits = wrapper.compute_tt_similarity(text, label_embeds, ml=True)
        logits_frame = wrapper.compute_tt_similarity_frame(text, label_embeds_frame, ml=True)
        loss_st = ranking_loss(logits, multi_hot_targets)
        loss_st_frame = ranking_loss(logits_frame, multi_hot_targets)
        optimizer.zero_grad()
        loss = loss_st + loss_st_frame
        loss.backward(create_graph=False)     
        optimizer.step()
        # lr_scheduler.step()

    return

# Load dataset
wrapper = CLAPWrapper(args)
# eval_dataset = FSD19_eval(root="../../../autodl-tmp/", download=False)
# train_dataset = FSD19_Text("./cap_fsd19.csv")
eval_dataset = DCASE19_val(root="../../../autodl-tmp/", download=False)
train_dataset = DCASE19_Text("./cap_dcase19.csv")

num_class = eval_dataset.class_num
prompt = ' {}'
y = [prompt.format(x) for x in eval_dataset.classes]

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

optimizer = torch.optim.AdamW([{'params': wrapper.prompts_embeddings.parameters()}, {'params': wrapper.prompts_embeddings_frame.parameters()}], lr=args.lr, betas=(0.9, 0.98), eps=1e-06)
# optimizer = torch.optim.SGD([{'params': wrapper.prompts_embeddings.parameters()}], lr=args.lr, momentum=0.9)
for n, p in wrapper.clap.named_parameters():
    if p.requires_grad == True:
        print(n)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader) * args.epochs)
# lr_scheduler = build_lr_scheduler(optimizer, args.epochs, 1, 1e-5)
# lr_scheduler = WarmupCosineSchedule(optimizer, len(train_dataloader) * 3, len(train_dataloader) * args.epochs)
device = "cuda" if args.use_cuda else "cpu"
# prompts_state_dict = torch.load("best_model.pt", map_location='cuda')
# wrapper.prompts_embeddings.load_state_dict(prompts_state_dict)
# prompts_state_dict = torch.load("best_model_frame.pt", map_location='cuda')
# wrapper.prompts_embeddings_frame.load_state_dict(prompts_state_dict)
best_map = 0
for e in range(args.epochs):
    print("Infer: ")
    mAP = eval_one_epoch(eval_dataset, wrapper, y=y)
    if mAP > best_map:
        best_map = mAP
        torch.save(wrapper.prompts_embeddings.state_dict(), 'best_model.pt')
        torch.save(wrapper.prompts_embeddings_frame.state_dict(), 'best_model_frame.pt')
    train_one_epoch(args, train_dataloader, wrapper, e, optimizer, lr_scheduler, device, num_class, y)
    # student

