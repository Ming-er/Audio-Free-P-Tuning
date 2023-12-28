from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import random
import torch.nn as nn
import torch
import torchaudio
import torchaudio.transforms as T

class AudioDataset(Dataset):
    def __init__(self, root: str, download: bool = False):
        self.root = os.path.expanduser(root)

    def __getitem__(self, index):
        raise NotImplementedError

    def download(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

class DCASE19_eval(AudioDataset):
    base_folder = 'DCASE2019/'
    audio_dir = 'eval_32k'
    label_col = 'event_label'
    file_col = 'filename'
    meta = {
        'filename': os.path.join('metadata','public.tsv'),
    }
    def __init__(self, root, reading_transformations: nn.Module = None, download: bool = True):
        super().__init__(root)
        self.classes = ["alarm bell ringing", "blender", "cat", "dishes", "dog", "electric shaver toothbrush", "frying", 
                        "running water", "speech", "vacuum cleaner"]
        self._load_meta()

        self.targets, self.audio_paths = [], []
        self.class_num = 10

        self.pre_transformations = reading_transformations
        for _, row in tqdm(self.df.iterrows()):
            file_path = os.path.join(self.root, self.base_folder, self.audio_dir, row[self.file_col])
            cls_names = row[self.label_col]
            cls_names = [cls.lower().replace('_',' ') for cls in cls_names]
            self.targets.append(cls_names)
            self.audio_paths.append(file_path)
            
    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        self.df = pd.read_csv(path, sep='\t')
        files_lst = list(set(self.df["filename"]))
        files_label_dic = {k: [] for k in files_lst}
        for i, row in tqdm(self.df.iterrows()):
            files_label_dic[row["filename"]].append(row["event_label"])
            files_label_dic[row["filename"]] = list(set(files_label_dic[row["filename"]]))
        self.df = pd.DataFrame(columns=["filename", "event_label"])
        for k in files_label_dic.keys():
            self.df = self.df.append({"filename": k, "event_label": files_label_dic[k]}, ignore_index=True)
        self.class_to_idx = {}
        self.idx_to_class = {}
        for i, category in enumerate(self.classes):
            self.class_to_idx[category] = i   
            self.idx_to_class[i] = category

    def load_audio_into_tensor(self, audio_path, audio_duration=10, resample=True):
        audio_time_series, sample_rate = torchaudio.load(audio_path)
        resample_rate = 44100
        
        if resample:
            resampler = T.Resample(sample_rate, resample_rate)
            audio_time_series = resampler(audio_time_series)
            sample_rate = resample_rate
            
        audio_time_series = audio_time_series.reshape(-1)

        if audio_duration * sample_rate >= audio_time_series.shape[0]:
            repeat_factor = int(np.ceil((audio_duration*sample_rate) / audio_time_series.shape[0]))
            audio_time_series = audio_time_series.repeat(repeat_factor)
            audio_time_series = audio_time_series[0 : audio_duration * sample_rate]
            
        else:
            start_index = random.randrange(audio_time_series.shape[0] - audio_duration * sample_rate)
            audio_time_series = audio_time_series[start_index:start_index + audio_duration * sample_rate] 
        return torch.FloatTensor(audio_time_series)
    
    def get_multihot(self, cls_name_lst):
        multihot_label = [0 for _ in range(10)]
        for cls in cls_name_lst:
            multihot_label[self.class_to_idx[cls]] = 1
        return multihot_label


    def __getitem__(self, index):
        file_path, target = self.audio_paths[index], self.targets[index]
        audio = self.load_audio_into_tensor(file_path, resample=True)
        multi_hot_target = torch.tensor(self.get_multihot(target))
        return audio, multi_hot_target

    def __len__(self):
        return len(self.audio_paths)
    
class DCASE19_val(AudioDataset):
    base_folder = 'DCASE2019/'
    audio_dir = 'validation_32k'
    label_col = 'event_label'
    file_col = 'filename'
    meta = {
        'filename': os.path.join('metadata','validation.tsv'),
    }
    def __init__(self, root, reading_transformations: nn.Module = None, download: bool = True):
        super().__init__(root)
        self.classes = ["alarm bell ringing", "blender", "cat", "dishes", "dog", "electric shaver toothbrush", "frying", 
                        "running water", "speech", "vacuum cleaner"]
        self._load_meta()

        self.targets, self.audio_paths = [], []
        self.class_num = 10

        self.pre_transformations = reading_transformations
        for i, row in tqdm(self.df.iterrows()):
            file_path = os.path.join(self.root, self.base_folder, self.audio_dir, row[self.file_col])
            cls_names = row[self.label_col]
            cls_names = [cls.lower().replace('_',' ') for cls in cls_names]
            self.targets.append(cls_names)
            self.audio_paths.append(file_path)
            
    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        self.df = pd.read_csv(path, sep='\t')
        files_lst = list(set(self.df["filename"]))
        files_label_dic = {k: [] for k in files_lst}
        for i, row in tqdm(self.df.iterrows()):
            if type(row["event_label"]) is str:
                files_label_dic[row["filename"]].append(row["event_label"])
                files_label_dic[row["filename"]] = list(set(files_label_dic[row["filename"]]))
        self.df = pd.DataFrame(columns=["filename", "event_label"])
        for k in files_label_dic.keys():
            if len(files_label_dic[k]) > 0:
                self.df = self.df.append({"filename": k, "event_label": files_label_dic[k]}, ignore_index=True)
        self.class_to_idx = {}
        self.idx_to_class = {}
        for i, category in enumerate(self.classes):
            self.class_to_idx[category] = i   
            self.idx_to_class[i] = category

    def load_audio_into_tensor(self, audio_path, audio_duration=10, resample=True):
        audio_time_series, sample_rate = torchaudio.load(audio_path)
        resample_rate = 44100
        
        if resample:
            resampler = T.Resample(sample_rate, resample_rate)
            audio_time_series = resampler(audio_time_series)
            sample_rate = resample_rate
            
        audio_time_series = audio_time_series.reshape(-1)

        if audio_duration * sample_rate >= audio_time_series.shape[0]:
            repeat_factor = int(np.ceil((audio_duration*sample_rate) / audio_time_series.shape[0]))
            audio_time_series = audio_time_series.repeat(repeat_factor)
            audio_time_series = audio_time_series[0 : audio_duration * sample_rate]
            
        else:
            start_index = random.randrange(audio_time_series.shape[0] - audio_duration * sample_rate)
            audio_time_series = audio_time_series[start_index:start_index + audio_duration * sample_rate] 
        return torch.FloatTensor(audio_time_series)
    
    def get_multihot(self, cls_name_lst):
        multihot_label = [0 for _ in range(10)]
        for cls in cls_name_lst:
            multihot_label[self.class_to_idx[cls]] = 1
        return multihot_label


    def __getitem__(self, index):
        file_path, target = self.audio_paths[index], self.targets[index]
        audio = self.load_audio_into_tensor(file_path, resample=True)
        multi_hot_target = torch.tensor(self.get_multihot(target))
        return audio, multi_hot_target

    def __len__(self):
        return len(self.audio_paths)

class   DCASE19_Text():
    def __init__(self, csv_path):
        super().__init__()
        self.cap_df = pd.read_csv(csv_path)
        self.classes = ["alarm bell ringing", "blender", "cat", "dishes", "dog", "electric shaver toothbrush", "frying", 
                        "running water", "speech", "vacuum cleaner"]
        
        self.texts = list(self.cap_df["caption"])
        self.targets = list(self.cap_df["class"])
        self.targets = [eval(tmp) for tmp in self.targets]
        self.class_to_idx = {}
        for i, category in enumerate(self.classes):
            self.class_to_idx[category] = i

    def get_multihot(self, idx_lst):
        multihot_label = [0 for _ in range(10)]
        for idx in idx_lst:
            multihot_label[idx] = 1
        return multihot_label
    
    def __getitem__(self, index):
        multi_hot_target = torch.tensor(self.get_multihot(self.targets[index]))
        return self.texts[index], multi_hot_target

    def __len__(self):
        return len(self.texts)