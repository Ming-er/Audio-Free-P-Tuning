from CLAPWrapper import CLAPWrapper
import torch
import numpy as np
from copy import deepcopy
import torch.nn as nn
from esc50_dataset import ESC50
from urban8k_dataset import URBAN8K
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from importlib_resources import files
from models.utils import read_config_as_args
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import Dataset, DataLoader
import math

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule

class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, dataset='esc'):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = 50 if dataset == 'esc' else 10
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
class ModelEma:
    def __init__(self, model, decay=0.999, device=""):
        # make a copy of the model for accumulating moving average of weights
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if device:
            self.ema.to(device=device)
        for p in self.ema.parameters():
            p.requires_grad_(False)


    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            for k, ema_v in self.ema.state_dict().items():
                model_v = msd[k].detach()
                if self.device:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(ema_v * self.decay + (1. - self.decay) * model_v)
    

class CLAP_Classifier(nn.Module):
    def __init__(self, norm_label_embeds):        
        super().__init__()
        self.classifier = nn.Linear(norm_label_embeds.size(1), norm_label_embeds.size(0), bias=False)
        self.classifier.weight = nn.Parameter(norm_label_embeds)   
        self.classifier.to(device)         
        
    def forward(self, audio_embeds):
        return self.classifier(audio_embeds)

def get_label_embeds(dataset, wrapper):
    prompt = 'this is a sound of '
    y = [prompt + x for x in dataset.classes]

    wrapper.clap.eval()
    label_embeds = wrapper.get_text_embeddings(y)
    norm_label_embeds = wrapper.norm_embeds(label_embeds)
    return norm_label_embeds

def eval_one_epoch(dataset, wrapper, clap_classifier, student=True):
    eval_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    y_preds, y_labels = [], []
    for i, (audio, _, one_hot_target) in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
        audio = audio.to(device, non_blocking=True)
        wrapper.clap.eval()
        audio_embeds = wrapper.norm_embeds(wrapper.get_audio_embeddings(audio))
        if student:
            clap_classifier.eval()
            logits = clap_classifier(audio_embeds)
        else:
            logits = clap_classifier.ema(audio_embeds)
        y_pred = F.softmax(logits.detach().cpu(), dim=1).numpy()
        y_preds.append(y_pred)
        y_labels.append(one_hot_target.detach().cpu().numpy())
    y_labels, y_preds = np.concatenate(y_labels, axis=0), np.concatenate(y_preds, axis=0)
    acc = accuracy_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))
    print('ESC50 Accuracy {}'.format(acc)) 
    # rpt = classification_report(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1), target_names=dataset.classes)
    # print(rpt) 

def train_one_epoch(args, train_dataloader, wrapper, ema_wrapper, clap_classifier, ema_clap_classifier, epoch, optimizer, scheduler, device, class_thresh):
    loss_func = SCELoss(alpha=1.0, beta=0.20)
    reliable_p_labels_num = 0
    class_ge_thresh_num = [0 for i in range(50)]
    for i, (audio, _, one_hot_targets) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        ema_clap_classifier.decay = args.ema_decay_warm + (args.ema_decay - args.ema_decay_warm) * min(1, i + epoch * len(train_dataloader) / args.warm_ema)
        ema_wrapper.ema_clap.decay = ema_clap_classifier.decay
        audio = audio.to(device)
        one_hot_targets = one_hot_targets.to(device)
        
        with torch.no_grad():         
            ema_embeds = ema_wrapper.norm_embeds(ema_wrapper.get_audio_embeddings(audio))    
            ema_temp = ema_wrapper.return_temp()
            probs_ema = F.softmax(ema_clap_classifier.ema(ema_embeds) * ema_temp, dim=-1)
            
            score, pseudo_targets = probs_ema.max(-1)
            _, targets = one_hot_targets.max(-1)
            # conf_threshold = cls_thresh[pseudo_targets].to(device)
            # conf_mask = score > conf_threshold
            if epoch == 0:
                conf_mask = score > args.conf_threshold
            else:
                conf_mask = score.ge(args.conf_threshold * class_thresh[pseudo_targets])
            # pseudo_label_acc = accuracy_score(np.argmax(one_hot_targets.detach().cpu().numpy(), axis=1), np.argmax(probs_ema.detach().cpu().numpy(), axis=1))  
            # print(pseudo_label_acc)
            for i in range(score.size(0)):
                if score[i] > args.conf_threshold:
                    # print(pseudo_targets[i])
                    class_ge_thresh_num[pseudo_targets[i]] += 1

        wrapper.clap.train()
        clap_classifier.train()
        embeds = wrapper.norm_embeds(wrapper.get_audio_embeddings(audio, is_spec_augment=True))
        
        logits = clap_classifier(embeds) * wrapper.return_temp()
        loss_st = loss_func(logits[conf_mask], pseudo_targets[conf_mask]) 

        probs = F.softmax(logits,dim=-1) 
        probs_avg = probs.mean(0) # average prediction probability across all gpus
        loss_fair = -(torch.log(probs_avg)).mean()   

        optimizer.zero_grad()
        loss = loss_st
        loss.backward(create_graph=False)     
        optimizer.step()

        ema_wrapper.ema_clap.update(wrapper.clap)
        ema_clap_classifier.update(clap_classifier)

        scheduler.step()
    for i in range(50):
        class_thresh[i] = class_ge_thresh_num[i] / max(class_ge_thresh_num)
    return class_thresh
# Load dataset
wrapper = CLAPWrapper(args)
dataset = ESC50(root="../../../autodl-tmp/", download=False)
# dataset = URBAN8K(root="../../../autodl-tmp/", download=False)

norm_label_embeds = get_label_embeds(dataset, wrapper)
clap_classifier = CLAP_Classifier(norm_label_embeds)

ema_clap_classifier = ModelEma(clap_classifier, args.ema_decay, device)
ema_wrapper = CLAPWrapper(args)
ema_wrapper.ema_clap = ModelEma(ema_wrapper.clap, args.ema_decay, device)
train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

optimizer = torch.optim.AdamW([{'params': wrapper.clap.parameters()}, {'params': clap_classifier.parameters()}], lr=args.lr, weight_decay=0.02, betas=(0.9, 0.98), eps=1e-06)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader) * args.epochs)
device = "cuda" if args.use_cuda else "cpu"
class_thresh = torch.zeros((50)).to(device)
for e in range(args.epochs):
    wrapper.clap.train()
    class_thresh = train_one_epoch(args, train_dataloader, wrapper, ema_wrapper, clap_classifier, ema_clap_classifier, e, optimizer, lr_scheduler, device, class_thresh)
    print(class_thresh)
    # student
    print("Student infer: ")
    eval_one_epoch(dataset, wrapper, clap_classifier)
    # teacher
    print("Teacher infer: ")
    eval_one_epoch(dataset, ema_wrapper, ema_clap_classifier, student=False)
