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

def predict(image : Image.Image):
    image = transform(image).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        logits = model(image)
        probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()

    results = {}

    for disease, probability in zip(label_columns, probabilities):
        results[disease] = float(probability)

    return results