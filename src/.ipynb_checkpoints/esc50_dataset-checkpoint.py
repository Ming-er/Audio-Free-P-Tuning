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


class ESC50(AudioDataset):
    base_folder = 'ESC-50-master'
    url = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
    filename = "ESC-50-master.zip"
    num_files_in_dir = 2000
    audio_dir = 'audio'
    label_col = 'category'
    file_col = 'filename'
    meta = {
        'filename': os.path.join('meta','esc50.csv'),
    }

    def __init__(self, root, reading_transformations: nn.Module = None, download: bool = True, support=None):
        super().__init__(root)
        self._load_meta()

        self.targets, self.audio_paths = [], []
        self.class_num = 50
        self.pre_transformations = reading_transformations
        print("Loading audio files")
        # self.df['filename'] = os.path.join(self.root, self.base_folder, self.audio_dir) + os.sep + self.df['filename']
        self.df['category'] = self.df['category'].str.replace('_',' ')
        self.support_cnt = [0 for _ in range(50)]
        for _, row in tqdm(self.df.iterrows()):
            file_path = os.path.join(self.root, self.base_folder, self.audio_dir, row[self.file_col])
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

class   ESC_Text():
    def __init__(self, csv_path):
        super().__init__()
        self.cap_df = pd.read_csv(csv_path)
        self.classes = ['airplane', 'breathing', 'brushing teeth', 'can opening', 'car horn',
        'cat', 'chainsaw', 'chirping birds', 'church bells', 'clapping', 'clock alarm',
        'clock tick', 'coughing', 'cow', 'crackling fire', 'crickets', 'crow', 'crying baby',
        'dog', 'door wood creaks', 'door wood knock', 'drinking sipping', 'engine', 'fireworks',
        'footsteps', 'frog', 'glass breaking', 'hand saw', 'helicopter', 'hen', 'insects', 'keyboard typing',
        'laughing', 'mouse click', 'pig', 'pouring water', 'rain', 'rooster', 'sea waves', 'sheep', 'siren',
        'sneezing', 'snoring', 'thunderstorm', 'toilet flush', 'train', 'vacuum cleaner', 'washing machine',
        'water drops', 'wind']
        
        self.texts = list(self.cap_df["caption"])
        self.targets = list(self.cap_df["class"])

        # self.texts = []
        # self.targets = []
        templates = [
            "this is a sound of {}",
            "a sound of {}",
            "a recording of a {}",
            "a poor quality recording of a {}",
            "a simulation of a {} sound",
            "a faint sound of the {}",
            "a low bitrate audio of the {}",
            "a digitally generated {} sound",
            "an echo of a {}",
            "a distorted audio of the {}",
            "a snippet of the {} sound",
            "the hard to hear {}",
            "a loud {} sound",
            "a noisy {} sound",
            "a soft {} sound",
            "the sound of my {}",
            "the short sound of {}",
            "a close-up recording of {}",
            "a mono recording of the {}",
            "a composition of the {}",
            "a clipped sound of {}",
            "a corrupted mp3 of {}",
            "a muffled sound of the {}",
            "a clear {} sound",
            "a good quality sound of {}",
            "a blend of {} sounds",
            "a doodle of a {} sound",
            "a close mic recording of the {}",
            "the sound of a large {}",
            "the sound of a nice {}",
            "a high quality recording of {}",
            "a plushie sound of {}",
            "the nice sound of {}",
            "the small sound of {}",
            "the weird sound of {}",
            "the large sound of {}",
            "a mono recording of {}",
            "the plushie sound of {}",
            "a live recording of {}",
            "the cool sound of {}",
            "the barely audible {} sound",
            "a high pitched {} sound",
            "a deep {} sound",
            "a muffled {} sound",
            "the {} sound in the distance",
            "the soothing {} sound",
            "an annoying {} sound",
            "a repetitive {} sound",
            "a fading {} ",
            "a crackling {} sound",
            "a rhythmic {} sound",
            "a continuous {} sound",
            "an intermittent {} sound",
            "a distorted {} sound",
            "a shrill {} sound"
        ]
        # for i in range(50):
        #     for t in templates:
        #         self.texts.append(t.format(self.classes[i]))
        #         self.targets.append(i)
                
        self.class_to_idx = {}
        for i, category in enumerate(self.classes):
            self.class_to_idx[category] = i

    def __getitem__(self, index):
        target = self.targets[index]
        idx = torch.tensor(target)
        one_hot_target = torch.zeros(len(self.classes)).scatter_(0, idx, 1)
        return self.texts[index], one_hot_target

    def __len__(self):
        return len(self.texts)


class   ESC_Text_fs():
    def __init__(self, csv_path, support):
        super().__init__()
        self.cap_df = pd.read_csv(csv_path)
        self.classes = ['airplane', 'breathing', 'brushing teeth', 'can opening', 'car horn',
        'cat', 'chainsaw', 'chirping birds', 'church bells', 'clapping', 'clock alarm',
        'clock tick', 'coughing', 'cow', 'crackling fire', 'crickets', 'crow', 'crying baby',
        'dog', 'door wood creaks', 'door wood knock', 'drinking sipping', 'engine', 'fireworks',
        'footsteps', 'frog', 'glass breaking', 'hand saw', 'helicopter', 'hen', 'insects', 'keyboard typing',
        'laughing', 'mouse click', 'pig', 'pouring water', 'rain', 'rooster', 'sea waves', 'sheep', 'siren',
        'sneezing', 'snoring', 'thunderstorm', 'toilet flush', 'train', 'vacuum cleaner', 'washing machine',
        'water drops', 'wind']
        templates = [
            "this is a sound of {}",
            "a sound of {}",
            "a recording of a {}",
            "the sound of {}",
            "a poor quality recording of a {}",
            "a simulation of a {} sound",
            "a faint sound of the {}",
            "a low bitrate audio of the {}",
            "a digitally generated {} sound",
            "an echo of a {}",
            "a distorted audio of the {}",
            "a snippet of the {} sound",
            "the hard to hear {}",
            "a loud {} sound",
            "a noisy {} sound",
            "a soft {} sound",
            "the sound of my {}",
            "the short sound of {}",
            "a close-up recording of {}",
            "a mono recording of the {}",
            "a composition of the {}",
            "a clipped sound of {}",
            "a corrupted mp3 of {}",
            "a muffled sound of the {}",
            "a clear {} sound",
            "a good quality sound of {}",
            "a blend of {} sounds",
            "a doodle of a {} sound",
            "a close mic recording of the {}",
            "the sound of a large {}",
            "the sound of a nice {}",
            "the weird sound of {}",
            "a high quality recording of {}",
            "a plushie sound of {}",
            "the nice sound of {}",
            "the small sound of {}",
            "the weird sound of {}",
            "the large sound of {}",
            "a long audio of {}",
            "a mono recording of {}",
            "the plushie sound of {}",
            "a live recording of {}",
            "the cool sound of {}",
            "the barely audible {} sound",
            "a high pitched {} sound",
            "a deep {} sound",
            "a muffled {} sound",
            "the {} sound in the distance",
            "sound of {} in the background",
            "the soothing {} sound",
            "an annoying sound of {}",
            "a repetitive {} sound",
            "a fading {} sound",
            "a crackling {} sound",
            "a rhythmic sound of {}",
            "a continuous {} sound",
            "an intermittent {} sound",
            "a distorted sound of {}",
            "a shrill {} sound",
            "{} is making sound",
            "{} sound in a video",
            "{} sound could be heard",
            "an audio of {}", 
        ]
        # self.texts = list(self.cap_df["caption"])
        # self.targets = list(self.cap_df["class"])
        self.texts = []
        self.targets = []
        for i in range(50):
            for t in templates:
                self.texts.append(t.format(self.classes[i]))
                self.targets.append(i)
        self.cnt_lst = [0 for _ in range (50)]
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