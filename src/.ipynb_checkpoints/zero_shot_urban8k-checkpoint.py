from CLAPWrapper import CLAPWrapper
from urban8k_dataset import URBAN8K
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from importlib_resources import files
from models.utils import read_config_as_args
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import Dataset, DataLoader

config_as_str = files('configs').joinpath('config.yml').read_text()
args = read_config_as_args(config_as_str, is_config_str=True)
use_frame = False
device = "cuda" if args.use_cuda else "cpu"

# Load dataset
eval_dataset = URBAN8K(root="../../../autodl-tmp/", download=False)
prompt = 'this is a sound of '
y = [prompt + x for x in eval_dataset.classes]

wrapper = CLAPWrapper(args)
wrapper.clap.eval()
label_embeds = wrapper.get_text_embeddings(y)
eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

y_preds, y_labels = [], []
for i, (audio, _, one_hot_target) in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
    audio = audio.to(device, non_blocking=True)
    similarity = wrapper.compute_at_similarity(audio, label_embeds)
    y_pred = F.softmax(similarity.detach().cpu(), dim=1).numpy()
    y_preds.append(y_pred)
    y_labels.append(one_hot_target.detach().cpu().numpy())

y_labels, y_preds = np.concatenate(y_labels, axis=0), np.concatenate(y_preds, axis=0)
acc = accuracy_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))
print('Urban8k Accuracy {}'.format(acc))
rpt = classification_report(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1), target_names=eval_dataset.classes)
print(rpt)


