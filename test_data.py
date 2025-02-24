import os
import torch
from PIL import Image
from torchvision import transforms
from PIL import Image
from model_data import model, device


model.load_state_dict(torch.load("cancerClassifier_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((50,50)),
    transforms.ToTensor(),
])

data_path = (r"C:\Users\samue\.cache\kagglehub\datasets\paultimothymooney\breast-histopathology-images\versions\1")
image_path = os.path.join(data_path, r"\10255\0\10255_idx5_x1051_y1051_class0.png  ")



img = Image.open(image_path).convert("RGB")
img = transform(img)
img = img.unsqueeze(0)

img = img.to(device) 

output = model(img)
prediction = torch.argmax(output, dim=1).item()
print("Prediction:", "Cancer" if prediction == 1 else "No Cancer")