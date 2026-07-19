import torch
from torchvision.transforms import transforms
from .model import model, DEVICE, label_columns
from PIL import Image
from .config import THRESHOLDS

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

def predict(image : Image.Image):
    image = image.convert("RGB")
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
