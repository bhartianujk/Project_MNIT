import io
import json

import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from torchvision import transforms
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(title="CIFAR-10 Animal Classifier API")
app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
)
class AnimalCnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


with open("saved_model_demo/class_names.json", "r") as f:
    idx_to_class = {int(k): v for k, v in json.load(f).items()}

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

model = AnimalCnn()
state_dict = torch.load("saved_model_demo/cifar10_animals.pt", weights_only=True)
model.load_state_dict(state_dict)
model.eval()


def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    return tensor


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file")

    try:
        image_bytes = await file.read()
        tensor = preprocess_image(image_bytes)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = int(torch.argmax(probs).item())
            confidence = float(probs[pred_idx].item())

        probabilities = {
            idx_to_class[i]: float(probs[i].item()) for i in range(len(idx_to_class))
        }

        return {
            "predicted_class": idx_to_class[pred_idx],
            "confidence": confidence,
            "probabilities": probabilities
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
