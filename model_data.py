import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir=root_dir
        self.transform = transform
        self.data = []

        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            for label in ["0", "1"]:
                label_path = os.path.join(folder_path, label)
                if os.path.isdir(label_path):
                    for img_name in os.listdir(label_path):
                        img_path = os.path.join(label_path, img_name)
                        self.data.append((img_path, int(label)))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


dataset_path= r"C:\Users\samue\.cache\kagglehub\datasets\paultimothymooney\breast-histopathology-images\versions\1"


transform = transforms.Compose([
    transforms.Resize((50,50)),
    transforms.ToTensor(),
])

dataset = CustomImageDataset(dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

first_batch = next(iter(dataloader))
print("Batch shape:", first_batch[0].shape, "Labels:", first_batch[1])
print("Batch shape:", first_batch[0].shape)

class CancerClassifier(nn.Module):
    def __init__(self):
        super(CancerClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3,32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32,64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128,3)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = CancerClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
print_interval = 100

for epoch in range(num_epochs):
    running_loss = 0.0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % print_interval == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    print(f"Epoch [{epoch+1}/{num_epochs}] Completed - Average Loss: {running_loss / len(dataloader):.4f}")

torch.save(model.state_dict(), "cancerClassifier_model.pth")