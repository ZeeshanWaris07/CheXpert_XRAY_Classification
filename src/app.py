from fastapi import FastAPI,File,UploadFile
from PIL import Image
from .inference import predict

app = FastAPI()

@app.get("/")
def Home():
    return {"message": "Chest Disease X-Ray API is running"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # Read the uploaded image
    image = Image.open(file.file)
    
    # Perform prediction
    result = predict(image)
    
    return {"prediction": result}
