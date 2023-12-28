from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import random
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
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

class URBAN8K(AudioDataset):
    base_folder = 'UrbanSound8K'
    audio_dir = 'audio'
    label_col = 'class'
    file_col = 'slice_file_name'
    meta = {
        'filename': os.path.join('metadata','UrbanSound8K.csv')
    }

    def __init__(self, root, reading_transformations: nn.Module = None, download: bool = True, support=None):
        super().__init__(root)
        self._load_meta()

        self.targets, self.audio_paths = [], []
        self.class_num = 10
        self.pre_transformations = reading_transformations
        print("Loading audio files")
        self.df["class"] = self.df["class"].str.replace('_',' ')
        self.support_cnt = [0 for _ in range(10)]

        for _, row in tqdm(self.df.iterrows()):
            file_path = os.path.join(self.root, self.base_folder, self.audio_dir, "fold" + str(row["fold"]), row[self.file_col])
            self.support_cnt[self.class_to_idx[row[self.label_col]]] += 1
            if support != None:
                if self.support_cnt[self.class_to_idx[row[self.label_col]]] <= support:
                    self.targets.append(row[self.label_col])
                    self.audio_paths.append(file_path)
            else:
                self.targets.append(row[self.label_col])
                self.audio_paths.append(file_path)

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        self.df = pd.read_csv(path)
        self.class_to_idx = {}
        self.classes = [x.replace('_',' ') for x in sorted(self.df[self.label_col].unique())]
        for i, category in enumerate(self.classes):
            self.class_to_idx[category] = i
            
    def load_audio_into_tensor(self, audio_path, audio_duration=5, resample=False):
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
    
    def __getitem__(self, index):
        file_path, target = self.audio_paths[index], self.targets[index]
        audio = self.load_audio_into_tensor(file_path, resample=True)
        idx = torch.tensor(self.class_to_idx[target])
        one_hot_target = torch.zeros(len(self.classes)).scatter_(0, idx, 1)
        return audio, target, one_hot_target

    def __len__(self):
        return len(self.audio_paths)

class URBAN8K_Text_fs():
    def __init__(self, csv_path, support):
        super().__init__()
        self.cap_df = pd.read_csv(csv_path)
        self.classes = ['air conditioner', 'car horn', 'children playing', 'dog bark',
                            'drilling','engine idling', 'gun shot', 'jackhammer', 'siren', 'street music']

        self.texts = list(self.cap_df["caption"])
        self.targets = list(self.cap_df["class"])
        self.cnt_lst = [0 for _ in range (10)]
        self.sel_texts = []
        self.sel_targets = []
        for i in range(len(self.texts)):
            if self.cnt_lst[self.targets[i]] <= support:
                self.sel_texts.append(self.texts[i])
                self.sel_targets.append(self.targets[i])
                self.cnt_lst[self.targets[i]] += 1
            
                
        self.class_to_idx = {}
        for i, category in enumerate(self.classes):
            self.class_to_idx[category] = i

    def __getitem__(self, index):
        target = self.sel_targets[index]
        idx = torch.tensor(target)
        one_hot_target = torch.zeros(len(self.classes)).scatter_(0, idx, 1)
        return self.sel_texts[index], one_hot_target

    def __len__(self):
        return len(self.sel_texts)