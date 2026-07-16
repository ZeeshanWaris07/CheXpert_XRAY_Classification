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
from torchvision.models import densenet121 , DenseNet121_Weights


BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

train_df["Path"] = train_df["Path"].str.replace(
    "CheXpert-v1.0-small/",
    "",
    regex=False
)

validation_df["Path"] = validation_df["Path"].str.replace(
    "CheXpert-v1.0-small/",
    "",
    regex=False
)

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
    

train_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = ChestXrayDataset(dataframe = train_df, root_dir = BASE_DIR / "data", label_cols = label_columns, transforms = train_transforms)
validation_dataset = ChestXrayDataset(dataframe = validation_df, root_dir = BASE_DIR / "data", label_cols = label_columns, transforms = val_transforms)

train_dataloader = DataLoader(train_dataset,BATCH_SIZE,shuffle = True)
val_dataloader = DataLoader(validation_dataset,BATCH_SIZE,shuffle = False)

weights = DenseNet121_Weights.DEFAULT
model = densenet121(weights = weights)

model.classifier = nn.Linear(in_features = model.classifier.in_features,out_features = len(label_columns))

model.to(DEVICE)

images,labels = next(iter(train_dataloader))
images = images.to(DEVICE)
labels = labels.to(DEVICE)

outputs = model(images)
print(outputs.shape)  # Should be [BATCH_SIZE, len(label_columns)]