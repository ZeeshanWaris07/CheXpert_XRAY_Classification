import torch

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
    "Support Devices",
]

THRESHOLDS = {
    "No Finding": 0.30,
    "Enlarged Cardiomediastinum": 0.00,
    "Cardiomegaly": 0.05,
    "Lung Opacity": 0.28,
    "Lung Lesion": 0.01,
    "Edema": 0.27,
    "Consolidation": 0.07,
    "Pneumonia": 0.04,
    "Atelectasis": 0.15,
    "Pneumothorax": 0.14,
    "Pleural Effusion": 0.39,
    "Pleural Other": 0.05,
    "Fracture": 0.50,
    "Support Devices": 0.35
}