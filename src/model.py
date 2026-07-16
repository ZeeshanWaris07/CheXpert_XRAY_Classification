import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch import nn
from PIL import Image


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

class ChestXrayDataset(Dataset):
    def __init__(self,dataframe,root_dir,label_cols,transforms = None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.label_cols = label_cols
        self.transform = transforms

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, key):
        row = self.dataframe.iloc[key]
        image_path = self.root_dir / row["Path"]
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        labels = row[self.label_cols].values.astype(np.float32)
        labels = torch.tensor(labels)

        return image,labels
    

