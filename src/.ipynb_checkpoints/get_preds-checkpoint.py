"""
This is an example using CLAP to perform zeroshot 
    classification on ESC50 (https://github.com/karolpiczak/ESC-50).
"""
import pandas as pd
from CLAPWrapper import CLAPWrapper
from esc50_dataset import ESC50
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# Load dataset
dataset = ESC50(root="data_path", download=False)
prompt = 'this is a sound of '
y = [prompt + x for x in dataset.classes]


# Load and initialize CLAP
weights_path = "../../CLAP_weights_2022.pth"
clap_model = CLAPWrapper(weights_path, use_cuda=False)


# Computing text embeddings
text_embeddings = clap_model.get_text_embeddings(y)

# Computing audio embeddings
y_preds, y_labels, file_names = [], [], []
for i in tqdm(range(len(dataset))):
    x, _, one_hot_target = dataset.__getitem__(i)
    audio_embeddings = clap_model.get_audio_embeddings([x], resample=True)
    similarity = clap_model.compute_similarity(audio_embeddings, text_embeddings)
    y_pred = F.softmax(similarity.detach().cpu(), dim=1).numpy()
    y_preds.append(y_pred)
    y_labels.append(one_hot_target.detach().cpu().numpy())
    file_name = x.split("/")[-1]
    file_names.append(file_name)

y_labels, y_preds = np.concatenate(y_labels, axis=0), np.concatenate(y_preds, axis=0)
preds_df = pd.DataFrame(y_preds)
preds_df.columns = dataset.classes
preds_df["filename"] = file_names
preds_df.to_csv("preds.csv")
acc = accuracy_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))
print('ESC50 Accuracy {}'.format(acc))

"""
The output:

ESC50 Accuracy: 82.6%

"""
