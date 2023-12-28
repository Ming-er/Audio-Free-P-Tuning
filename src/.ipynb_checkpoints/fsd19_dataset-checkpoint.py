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

class FSD19_eval(AudioDataset):
    base_folder = 'FSD2019/'
    audio_dir = 'FSDKaggle2019_audio_test'
    label_col = 'labels'
    file_col = 'fname'
    meta = {
        'filename': os.path.join('FSDKaggle2019_meta','test_post_competition.csv'),
        'voca_filename': os.path.join('FSDKaggle2019_meta','vocabulary.csv')
    }
    def __init__(self, root, reading_transformations: nn.Module = None, download: bool = True):
        super().__init__(root)
        self._load_meta()

        self.targets, self.audio_paths = [], []
        self.class_num = 80
        self.pre_transformations = reading_transformations
        for i, row in tqdm(self.df.iterrows()):
            file_path = os.path.join(self.root, self.base_folder, self.audio_dir, row[self.file_col])
            cls_names = row[self.label_col].split(",")
            cls_names = [cls.lower().replace('_and_',', ').replace('_', ' ') for cls in cls_names]
            self.targets.append(cls_names)
            self.audio_paths.append(file_path)
            
    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        self.df = pd.read_csv(path, sep=',')
        self.class_to_idx = {}
        self.idx_to_class = {}
        voca_path = os.path.join(self.root, self.base_folder, self.meta['voca_filename'])
        self.voca_df = pd.read_csv(voca_path, sep=',', header=None)
        self.classes = self.voca_df[1].tolist()
        self.classes = [cls.lower().replace('_and_',', ').replace('_', ' ') for cls in self.classes]
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
        multihot_label = [0 for _ in range(80)]
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

class   FSD19_Text():
    def __init__(self, csv_path):
        super().__init__()
        self.cap_df = pd.read_csv(csv_path)
        self.classes = ['accelerating, revving, vroom', 'accordion', 'acoustic guitar', 'applause', 'bark', 'bass drum', 'bass guitar', 'bathtub (filling or washing)', 'bicycle bell', 'burping, eructation', 'bus', 'buzz', 'car passing by', 'cheering', 'chewing, mastication', 'child speech, kid speaking', 'chink, clink', 'chirp, tweet', 'church bell', 'clapping', 'computer keyboard', 'crackle', 'cricket', 'crowd', 'cupboard open or close', 'cutlery, silverware', 'dishes, pots, pans', 'drawer open or close', 'drip', 'electric guitar', 'fart', 'female singing', 'female speech, woman speaking', 'fill (with liquid)', 'finger snapping', 'frying (food)', 'gasp', 'glockenspiel', 'gong', 'gurgling', 'harmonica', 'hi-hat', 'hiss', 'keys jangling', 'knock', 'male singing', 'male speech, man speaking', 'marimba, xylophone', 'mechanical fan', 'meow', 'microwave oven', 'motorcycle', 'printer', 'purr', 'race car, auto racing', 'raindrop', 'run', 'scissors', 'screaming', 'shatter', 'sigh', 'sink (filling or washing)', 'skateboard', 'slam', 'sneeze', 'squeak', 'stream', 'strum', 'tap', 'tick-tock', 'toilet flush', 'traffic noise, roadway noise', 'trickle, dribble', 'walk, footsteps', 'water tap, faucet', 'waves, surf', 'whispering', 'writing', 'yell', 'zipper (clothing)']
        
        self.texts = list(self.cap_df["caption"])
        self.targets = list(self.cap_df["class"])
        self.targets = [eval(tmp) for tmp in self.targets]
        self.class_to_idx = {}
        for i, category in enumerate(self.classes):
            self.class_to_idx[category] = i

    def get_multihot(self, idx_lst):
        multihot_label = [0 for _ in range(80)]
        for idx in idx_lst:
            multihot_label[idx] = 1
        return multihot_label
    
    def __getitem__(self, index):
        multi_hot_target = torch.tensor(self.get_multihot(self.targets[index]))
        return self.texts[index], multi_hot_target

    def __len__(self):
        return len(self.texts)

