from CLAPWrapper import CLAPWrapper
import torch
import numpy as np
from copy import deepcopy
import torch.nn as nn
from esc50_dataset import *
from urban8k_dataset import *
from vggsound_dataset import *
from tut_dataset import *
from gtzan_dataset import *
from vs_dataset import *
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from importlib_resources import files
from models.utils import read_config_as_args
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
import math

import torch
from torch.optim.lr_scheduler import _LRScheduler

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class _BaseWarmupScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        last_epoch=-1,
        verbose=False
    ):
        self.successor = successor
        self.warmup_epoch = warmup_epoch
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epoch:
            self.successor.step(epoch)
            self._last_lr = self.successor.get_last_lr()
        else:
            super().step(epoch)


class ConstantWarmupScheduler(_BaseWarmupScheduler):
    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        cons_lr,
        last_epoch=-1,
        verbose=False
    ):
        self.cons_lr = cons_lr
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        return [self.cons_lr for _ in self.base_lrs]

def build_lr_scheduler(optimizer, max_epoch, warmup_epoch, warmup_lr):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(max_epoch)
    )
    scheduler.last_epoch = warmup_epoch
    scheduler = ConstantWarmupScheduler(
        optimizer, scheduler, warmup_epoch,
        warmup_lr
    )
    return scheduler

class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_class):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_class
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=-1)
        # print(pred.size())
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        # print(label_one_hot.size())
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))
        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss

config_as_str = files('configs').joinpath('config.yml').read_text()
args = read_config_as_args(config_as_str, is_config_str=True)
device = "cuda" if args.use_cuda else "cpu"
# cls_thresh = torch.tensor([0.9655928, 0.9258593, 0.96030074, 0.5713826, 0.992004, 0.9807508, 0.9956363, 0.9493885, 0.99988043, 0.9837974, 0.94677323, 0.95169973, 0.9236182, 0.67578864, 0.9559114, 0.7783314, 0.46875516, 0.9982204, 0.983537, 0.99718755, 0.99377406, 0.29247752, 0.9671271, 0.9981565, 0.76083744, 0.97456837, 0.99933004, 0.99082243, 0.9157709, 0.9177454, 0.67059726, 0.95356333, 0.7654687, 0.64932483, 0.951495, 0.7368026, 0.8490892, 0.9026923, 0.9970534, 0.8739537, 0.98682004, 0.9717216, 0.987727, 0.99654084, 0.990991, 0.94827646, 0.9941754, 0.58936894, 0.9780376, 0.23703259])

def eval_one_epoch(dataset, wrapper, y):
    eval_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    wrapper.clap.eval()
    wrapper.prompts_embeddings.eval()
    label_embeds, _ = wrapper.get_text_embeddings(y)   
    y_preds, y_labels = [], []
    for i, (audio, _, one_hot_target) in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
        audio = audio.to(device, non_blocking=True)
        with torch.no_grad():    
            logits = wrapper.compute_at_similarity(audio, label_embeds, is_spec_augment=False)
            y_pred = F.softmax(logits.detach().cpu(), dim=1).numpy()
        y_preds.append(y_pred)
        y_labels.append(one_hot_target.detach().cpu().numpy())
    y_labels, y_preds = np.concatenate(y_labels, axis=0), np.concatenate(y_preds, axis=0)
    acc = accuracy_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))
    print('Accuracy {}'.format(acc)) 
    rpt = classification_report(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1), target_names=dataset.classes)
    print(rpt) 
    # rpt = confusion_matrix(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))
    # print(rpt)
    # cm = confusion_matrix(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))
    # for i in range(50):
    #     for j in range(50):
    #         if i != j and cm[i][j] !=0:
    #             print(dataset.classes[i], dataset.classes[j], cm[i][j])
    # print(cm)
    return acc
    # rpt = classification_report(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1), target_names=eval_dataset.classes)
    # print(rpt)
    # rpt = classification_report(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1), target_names=dataset.classes)
    # print(rpt) 

def train_one_epoch(args, train_dataloader, wrapper, epoch, optimizer, lr_scheduler, device, num_class, y, original_y):
    loss_func = SCELoss(alpha=1.0, beta=0.0, num_class=num_class)
    loss_func_kg = torch.nn.MSELoss()
    wrapper.clap.train()
    wrapper.frozen_caption_encoder.eval()
    wrapper.prompts_embeddings.train()
    y_preds, y_labels = [], []
    for i, (text, one_hot_targets) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        one_hot_targets = one_hot_targets.to(device)
        label_embeds, _ = wrapper.get_text_embeddings(y)
        norm_label_embeds = wrapper.norm_embeds(label_embeds)
        ori_label_embeds = wrapper.get_original_text_embedding(original_y)
        norm_ori_label_embeds = wrapper.norm_embeds(ori_label_embeds)
        logits = wrapper.compute_tt_similarity(text, label_embeds)
        _, targets = one_hot_targets.max(-1)
        loss_st = loss_func(logits, targets) 
        loss_kg = loss_func_kg(norm_label_embeds.squeeze(0), norm_ori_label_embeds)
        y_pred = F.softmax(logits.detach().cpu(), dim=1).numpy()
        y_preds.append(y_pred)
        y_labels.append(one_hot_targets.detach().cpu().numpy())
        optimizer.zero_grad()
        loss = loss_st
        loss.backward(create_graph=False)     
        optimizer.step()
        # lr_scheduler.step()
    y_labels, y_preds = np.concatenate(y_labels, axis=0), np.concatenate(y_preds, axis=0)
    acc = accuracy_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))
    print('Accuracy {}'.format(acc)) 
    return

# Load dataset
wrapper = CLAPWrapper(args)
# eval_dataset = GTZAN(root="../../../autodl-tmp/", download=False)
# train_dataset = GTZAN_Text("./cap_gtzan.csv")

eval_dataset = ESC50(root="../../../autodl-tmp/", download=False)
# train_dataset = ESC_Text("./cap_esc.csv")
train_dataset = ESC_Text_fs("./cap_esc.csv", support=2)

# eval_dataset = URBAN8K(root="../../../autodl-tmp/", download=False)
# train_dataset = URBAN8K_Text("./cap_urbansound8k.csv")
# train_dataset = URBAN8K_Text_fs("./cap_urbansound8k.csv", support=64)

# eval_dataset = VGGSoundTest(root="../../../autodl-tmp/", download=False)
# train_dataset = VGG_Text("./cap_vggsound.csv", eval_dataset.classes)

# eval_dataset = TUT_eval(root="../../../autodl-tmp/", download=False)
# train_dataset = TUT_Text("./cap_tut.csv", eval_dataset.classes)

# eval_dataset = VS_test(root="../../../autodl-tmp/", download=False)
# train_dataset = VS_Text("./cap_vs_3.csv", eval_dataset.classes)

num_class = eval_dataset.class_num
prompt = ' {}'
original_prompt = 'this is a sound of {}'
y = [prompt.format(x) for x in eval_dataset.classes]
original_y = [original_prompt.format(x) for x in eval_dataset.classes]

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

# optimizer = torch.optim.AdamW([{'params': wrapper.prompts_embeddings.parameters()}, {'params': wrapper.clap.caption_encoder.parameters()}], lr=args.lr, weight_decay=0.0, betas=(0.9, 0.98), eps=1e-06)
optimizer = torch.optim.SGD([{'params': wrapper.prompts_embeddings.parameters()}], lr=args.lr, momentum=0.9)
for n, p in wrapper.clap.named_parameters():
    if p.requires_grad == True:
        print(n)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader) * args.epochs)
# lr_scheduler = build_lr_scheduler(optimizer, args.epochs, 1, 1e-5)
device = "cuda" if args.use_cuda else "cpu"

best_acc = 0
for e in range(args.epochs):
    train_one_epoch(args, train_dataloader, wrapper, e, optimizer, lr_scheduler, device, num_class, y, original_y)
    # student
    print("Infer: ")
    acc = eval_one_epoch(eval_dataset, wrapper, y=y)
    if acc > best_acc:
        best_acc = acc
        torch.save(wrapper.prompts_embeddings.state_dict(), 'best_model.pt')
