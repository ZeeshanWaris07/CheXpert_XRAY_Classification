from torchvision.models import densenet121
import torch

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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 

model = densenet121(weights=None)
model.classifier = torch.nn.Linear(in_features=1024,out_features = len(label_columns))

checkpoint = torch.load("../weights/best_densenet_checkpoint.pth",map_location = DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])

model.to(DEVICE)