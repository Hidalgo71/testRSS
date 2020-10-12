# import cd as cd
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import wandb
from simpletransformers.classification import ClassificationModel

import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_df = pd.read_csv('E:/7allV03.csv', encoding='utf-8', header=None, names=['cat', 'text'])
print(train_df.head())
#print(train_df.cat.unique())
#print("Total categories", len(train_df.cat.unique()))
