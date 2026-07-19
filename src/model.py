from torchvision.models import densenet121
import torch
from .config import DEVICE, label_columns

model = densenet121(weights=None)
model.classifier = torch.nn.Linear(in_features=1024,out_features = len(label_columns))

checkpoint = torch.load("weights/best_densenet_checkpoint.pth",map_location = DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])

model.to(DEVICE)