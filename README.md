# PCBnet
resnet-based tool for analyzing PCBs

<!-- PROJECT LOGO -->
<br />
<div align="left">

<h3 align="center">PCB analyzer </h3>

<!-- ABOUT THE PROJECT -->
## About this Project

coming soon
 
## Prerequisite

Install [Python3](https://www.python.org/downloads/) on your computer.

Enter this into your computer's command line interface (terminal, control panel, etc.) to check the version:

  ```sh
  python --version
  ```

If the first number is not a 3, update to Python3.

## Setup

### Step 1: Download the repository

Download to your computer. 

### Step 2: Create a virtual environment: 
The default size limit on PyPI is 60MB.
Therefore, we will have to take the **virtual environment** route.

Create a virtual environment called *env* inside the working directory.

```sh
python3 -m venv env
```

Then, activate the virtual environment.


```sh
source env/bin/activate
```

### Step 3: Install requirements.txt

Avoid "dependency hell" by installing specific software versions known to work well together.

  ```sh
pip3 install -r requirements.txt
  ```

<!-- LICENSE -->
## Workflow

### 1. Annotate Images with LabelImg
  - Install using pip
  ```pip install labelImg
  ```


### 2. Parse the annotations to extract:
 - Bounding boxes: [xmin, ymin, xmax, ymax]
 - Labels: Class IDs.


### 3. Organize Dataset:
      ```
      dataset/
        images/
          img1.jpg
          img2.jpg
        annotations/
          img1.xml
          img2.xml
      ```
      
### 4. Dataset Class

```
import os
import xml.etree.ElementTree as ET
from PIL import Image
import torch

class PcbDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, annotation_dir, transforms=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transforms = transforms
        self.image_files = sorted(os.listdir(image_dir))
        self.annotation_files = sorted(os.listdir(annotation_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")
        
        # Load annotation
        annotation_path = os.path.join(self.annotation_dir, self.annotation_files[idx])
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        for obj in root.findall("object"):
            label = obj.find("name").text
            labels.append(int(label))  # Convert to class ID
            
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}
        
        if self.transforms:
            image = self.transforms(image)

        return image, target
```

### 5. Model Architecture
```
import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet50

class ResNetObjectDetector(nn.Module):
    def __init__(self, num_classes):
        super(ResNetObjectDetector, self).__init__()
        # Load ResNet backbone
        self.backbone = resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Remove final layers

        # Classification head
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        
        # Regression head
        self.reg_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 4)  # 4 coordinates: [xmin, ymin, xmax, ymax]
        )

    def forward(self, x):
        features = self.backbone(x)
        cls_logits = self.cls_head(features)
        bbox_preds = self.reg_head(features)
        return cls_logits, bbox_preds
```


### 6. Training Loop
  ```
  # Define loss functions
  cls_loss_fn = nn.CrossEntropyLoss()
  bbox_loss_fn = nn.SmoothL1Loss()
  
  # Optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
  
  # Training loop
  for epoch in range(num_epochs):
      model.train()
      for images, targets in train_loader:
          images = images.to(device)
          cls_targets = targets['labels'].to(device)  # Class labels
          bbox_targets = targets['boxes'].to(device)  # Bounding boxes
  
          optimizer.zero_grad()
          cls_logits, bbox_preds = model(images)
          
          # Compute losses
          cls_loss = cls_loss_fn(cls_logits, cls_targets)
          bbox_loss = bbox_loss_fn(bbox_preds, bbox_targets)
          loss = cls_loss + bbox_loss
          
          loss.backward()
          optimizer.step()
          
      print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
  ```

### 7. Inference

```
from torchvision.ops import nms

model.eval()
with torch.no_grad():
    images = images.to(device)
    cls_logits, bbox_preds = model(images)
    
    # Apply NMS
    scores, labels = cls_logits.max(dim=1)
    keep = nms(bbox_preds, scores, iou_threshold=0.5)
    final_boxes = bbox_preds[keep]
    final_labels = labels[keep]
```


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
