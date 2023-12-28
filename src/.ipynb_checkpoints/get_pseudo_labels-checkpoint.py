import pandas as pd
import numpy as np
df = pd.read_csv("preds.csv")
data = df.values[:, 1:]
cls_num = 8
retrieve_filenames = []
retrieve_cls = []
for i in range(50):
    max_idx_i = np.argsort(data[:, i])[-cls_num:]
    max_filename_i = data[:, -1][max_idx_i]
    for j in range(cls_num):
        retrieve_filenames.append(max_filename_i[j])
        retrieve_cls.append(df.columns[i + 1])
pseudo_labels_df = pd.DataFrame()
pseudo_labels_df["filename"] = retrieve_filenames
pseudo_labels_df["category"] = retrieve_cls
pseudo_labels_df.to_csv("pseudo_labels.csv", index=False)
    