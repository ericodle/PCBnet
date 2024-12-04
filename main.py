import os
import json
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# Dataset Definition
class PcbDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transforms=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transforms = transforms
        self.image_files = sorted(os.listdir(image_dir))
        self.annotation_files = sorted(os.listdir(annotation_dir))
        self.label_mapping = {
            "main_processor": 0, "trch": 1, "sram": 2, "nor_flash": 3,
            "fram": 4, "thermal_sensor": 5, "resistor": 6
        }

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, self.image_files[idx])).convert("RGB")
        with open(os.path.join(self.annotation_dir, self.annotation_files[idx])) as f:
            annotation = json.load(f)

        boxes, labels = [], []
        for obj in annotation["shapes"]:
            labels.append(self.label_mapping[obj["label"]])
            points = obj["points"]
            xmin, ymin = min(p[0] for p in points), min(p[1] for p in points)
            xmax, ymax = max(p[0] for p in points), max(p[1] for p in points)
            boxes.append([xmin, ymin, xmax, ymax])

        # Convert to tensors
        image = self.transforms(image) if self.transforms else ToTensor()(image)
        return image, {"boxes": torch.tensor(boxes, dtype=torch.float32), "labels": torch.tensor(labels, dtype=torch.int64)}


# Collate Function for Variable-Sized Targets
def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


# Data Preparation
def prepare_dataloaders(image_dir, annotation_dir, batch_size):
    transforms = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = PcbDataset(image_dir=image_dir, annotation_dir=annotation_dir, transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return dataloader


# Model Setup
def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# Training Loop
def train_model(model, train_loader, device, num_epochs, lr):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    return model


# Main Script
if __name__ == "__main__":
    # Paths to datasets
    train_image_dir = "./imgs/"
    train_annotation_dir = "./annotations/"

    # Hyperparameters
    num_classes = 7 + 1  # Add 1 for background class
    batch_size = 4
    num_epochs = 10
    learning_rate = 1e-4

    # Prepare data loaders
    train_loader = prepare_dataloaders(train_image_dir, train_annotation_dir, batch_size)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes)

    # Train model
    model = train_model(model, train_loader, device, num_epochs, learning_rate)

    # Save the model
    torch.save(model.state_dict(), "faster_rcnn_resnet18.pth")
    print("Model saved as 'faster_rcnn_resnet18.pth'")

