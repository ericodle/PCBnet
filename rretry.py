import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader

# Step 1: CNN Backbone (using a pre-trained ResNet for feature extraction)
class CNNBackbone(nn.Module):
    def __init__(self):
        super(CNNBackbone, self).__init__()
        # Use a pre-trained ResNet model, stripping the fully connected layers.
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove the FC layers
    
    def forward(self, x):
        return self.backbone(x)

# Step 2: Region Proposal Network (RPN)
class RPN(nn.Module):
    def __init__(self, in_channels=2048):  # ResNet50 output channels
        super(RPN, self).__init__()
        # Use a convolution layer to generate region proposals
        self.conv = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.cls_score = nn.Conv2d(256, 2, kernel_size=1)  # Object vs Background (binary classification)
        self.bbox_pred = nn.Conv2d(256, 4, kernel_size=1)  # 4 coordinates for bounding box
    
    def forward(self, x):
        x = self.conv(x)
        cls_score = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return cls_score, bbox_pred

# Step 3: Object Detection Head (final classification and bounding box regression)
class ObjectDetectionHead(nn.Module):
    def __init__(self, num_classes=21):  # Example: 21 classes (1 background + 20 object types)
        super(ObjectDetectionHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten for the fully connected layers
        return self.fc(x)

# Final Model Combining Backbone, RPN, and Object Detection Head
class ObjectDetectionModel(nn.Module):
    def __init__(self, num_classes=21):
        super(ObjectDetectionModel, self).__init__()
        self.backbone = CNNBackbone()
        self.rpn = RPN()
        self.head = ObjectDetectionHead(num_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        rpn_cls_score, rpn_bbox_pred = self.rpn(features)
        detection_output = self.head(features)
        return rpn_cls_score, rpn_bbox_pred, detection_output


import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# Custom Dataset
class ObjectDetectionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with all the subdirectories containing images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Loop through all subdirectories (representing the labels)
        for label in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label)
            
            if os.path.isdir(label_path):  # Ensure it's a directory
                # Loop through each image in the label directory
                for img_name in os.listdir(label_path):
                    img_path = os.path.join(label_path, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(label)  # Use subdirectory name as the label

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Open image and apply transformations
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        # Dummy bounding boxes (just for demonstration)
        # You should replace this with real bounding box annotations for object detection.
        # Format: [xmin, ymin, xmax, ymax]
        bounding_boxes = np.array([[50, 50, 150, 150]])  # Example, replace with real data
        
        # Assuming label is a string (like 'from_torchvision_ops'), we can convert it to an integer index
        label_idx = int(label == 'main_processor')  # Example: 0 for 'from_torchvision_ops', 1 for 'nor_flash'
        
        sample = {'image': image, 'bounding_boxes': bounding_boxes, 'label': label_idx}
        return sample

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Instantiate the dataset and DataLoader
dataset = ObjectDetectionDataset(root_dir='path/to/your/dataset', transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)


# Instantiate the model
model = ObjectDetectionModel(num_classes=21)  # 20 classes + 1 background


# Define the loss function
class ObjectDetectionLoss(nn.Module):
    def __init__(self):
        super(ObjectDetectionLoss, self).__init__()
        self.classification_loss = nn.CrossEntropyLoss()  # For classification
        self.bbox_loss = nn.SmoothL1Loss()  # For bounding box regression

    def forward(self, rpn_cls_score, rpn_bbox_pred, detection_output, targets):
        # Unpack targets (assuming targets are a list of dictionaries for each image)
        rpn_target_cls, rpn_target_bbox, detection_target_class = targets
        
        # Calculate RPN classification loss
        rpn_cls_loss = self.classification_loss(rpn_cls_score, rpn_target_cls)
        
        # Calculate RPN bounding box regression loss
        rpn_bbox_loss = self.bbox_loss(rpn_bbox_pred, rpn_target_bbox)
        
        # Calculate detection classification loss
        detection_cls_loss = self.classification_loss(detection_output, detection_target_class)
        
        return rpn_cls_loss + rpn_bbox_loss + detection_cls_loss

# Instantiate the loss function
loss_fn = ObjectDetectionLoss()


# Example training loop
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Assuming DataLoader and dataset are prepared
for epoch in range(num_epochs):
    model.train()
    for images, targets in train_loader:
        optimizer.zero_grad()
        
        # Forward pass
        rpn_cls_score, rpn_bbox_pred, detection_output = model(images)
        
        # Compute loss
        loss = loss_fn(rpn_cls_score, rpn_bbox_pred, detection_output, targets)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# Example inference
model.eval()
with torch.no_grad():
    images = some_test_images
    rpn_cls_score, rpn_bbox_pred, detection_output = model(images)
    
    # Apply NMS here to filter overlapping boxes
    # Use the detected bounding boxes and their class predictions for final output


from torchvision.ops import nms

def apply_nms(boxes, scores, iou_threshold=0.5):
    return nms(boxes, scores, iou_threshold)

# Example usage
nms_indices = apply_nms(predicted_boxes, predicted_scores)


