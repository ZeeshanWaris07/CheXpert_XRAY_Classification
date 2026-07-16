import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

label_columns = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices"
]

BASE_DIR = Path(__file__).resolve().parent.parent

train_df = pd.read_csv("../data/train.csv")
validation_df = pd.read_csv("../data/valid.csv")

train_df[label_columns] = train_df[label_columns].fillna(0)
validation_df[label_columns] = validation_df[label_columns].fillna(0)
train_df[label_columns] = train_df[label_columns].replace(-1, 0)
validation_df[label_columns] = validation_df[label_columns].replace(-1, 0)

print(train_df.head())
print(train_df.info())