# 🩻 Chest Disease Detection from Chest X-Rays using DenseNet121

A deep learning project for **multi-label chest disease classification** using **PyTorch** and **DenseNet121** fine-tuned on the **CheXpert** dataset.

The model predicts the probability of **14 thoracic diseases** from a single chest X-ray image and exposes the predictions through a **FastAPI backend** for easy deployment.

---

## Features

- Transfer Learning using pretrained DenseNet121
- Multi-label classification (14 diseases)
- Fine-tuning on the CheXpert dataset
- Automatic mixed precision (AMP) training
- Best model checkpoint saving
- Learning rate scheduling
- Data augmentation
- Threshold optimization for each disease
- FastAPI backend for inference
- Ready for frontend integration

---

## Diseases Detected

- No Finding
- Enlarged Cardiomediastinum
- Cardiomegaly
- Lung Opacity
- Lung Lesion
- Edema
- Consolidation
- Pneumonia
- Atelectasis
- Pneumothorax
- Pleural Effusion
- Pleural Other
- Fracture
- Support Devices

---

## Dataset

This project uses the **CheXpert** dataset released by Stanford University.

Dataset:
https://stanfordmlgroup.github.io/competitions/chexpert/

For training on Kaggle:

https://www.kaggle.com/datasets/ashery/chexpert

---

## Model Architecture

Backbone:

- DenseNet121 (ImageNet pretrained)

Classifier:

- Fully Connected Layer
- Output neurons: 14
- Sigmoid activation during inference

Loss Function:

- BCEWithLogitsLoss

Optimizer:

- AdamW

Learning Rate Scheduler:

- ReduceLROnPlateau

Mixed Precision:

- torch.amp.autocast()
- GradScaler

---

## Training Strategy

### Stage 1

- Load pretrained DenseNet121
- Freeze feature extractor
- Train only the classifier head

### Stage 2

- Load checkpoint
- Unfreeze all layers
- Fine-tune the complete network with a lower learning rate

---

## Data Preprocessing

Images are:

- Resized to 224×224
- Converted to RGB
- Normalized using ImageNet statistics

Training augmentations:

- Random Horizontal Flip
- Random Rotation

---

## Evaluation

Metrics used:

- ROC-AUC
- Precision
- Recall
- F1 Score
- Classification Report

Instead of using a fixed threshold of **0.5**, optimal thresholds were selected individually for each disease by maximizing the validation F1-score.

---

## Project Structure

```
ChestDiseaseX-Ray/

│
├── backend/
│   ├── src/
│   │   ├── app.py
│   │   ├── inference.py
│   │   ├── model.py
│   │   ├── config.py
│   │   └── __init__.py
│   │
│   └── weights/
│       └── best_densenet_checkpoint.pth
│
├── notebooks/
│   ├── training.ipynb
│   └── evaluation.ipynb
│
├── requirements.txt
└── README.md
```

---

## Running the Backend

Install dependencies

```bash
pip install -r requirements.txt
```

Start FastAPI

```bash
uvicorn src.app:app --reload
```

API will be available at

```
http://127.0.0.1:8000
```

Interactive API documentation

```
http://127.0.0.1:8000/docs
```

---

## Inference Pipeline

1. User uploads a chest X-ray.
2. Image is preprocessed.
3. DenseNet121 predicts logits.
4. Sigmoid converts logits into probabilities.
5. Disease-specific thresholds are applied.
6. Predicted diseases and probabilities are returned as JSON.

---

## Example Response

```json
{
  "predictions": [
    {
      "disease": "Lung Opacity",
      "probability": 0.91
    },
    {
      "disease": "Pleural Effusion",
      "probability": 0.82
    }
  ]
}
```

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- FastAPI
- NumPy
- Pandas
- Pillow
- Matplotlib
- Scikit-learn
- Uvicorn
- Kaggle

---

## Future Improvements

- Vision Transformer (ViT)
- ConvNeXt
- EfficientNetV2
- Class imbalance handling using Focal Loss
- WeightedRandomSampler
- Explainability using Grad-CAM
- Docker deployment
- React frontend
- Cloud deployment (Render/AWS)

---

## Disclaimer

This project is intended **for educational and research purposes only**.

The predictions produced by the model **must not be used for clinical diagnosis or medical decision-making**.

---

## Author

**Zeeshan Waris**

If you found this project interesting, feel free to ⭐ the repository.
