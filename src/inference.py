import torch
from torchvision.transforms import transforms
from model import model, DEVICE, label_columns
from PIL import Image


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

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

def predict(image : Image.Image):
    image = transform(image).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        logits = model(image)
        probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()

    detected = []

    results = {}

    for disease, prob in zip(label_columns, probabilities):

        probability = float(prob)
        threshold = THRESHOLDS[disease]

        prediction = probability >= threshold

        results[disease] = {
            "probability": round(probability, 4),
            "prediction": prediction
        }

        if prediction:
            detected.append(disease)

    return {
        "detected_diseases": detected,
        "results": results
    }
