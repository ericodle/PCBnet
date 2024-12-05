import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Original label mapping from the dataset
label_mapping = {
    "main_processor": 0, "trch": 1, "sram": 2, "nor_flash": 3,
    "fram": 4, "thermal_sensor": 5, "resistor": 6
}
label_names = {v: k for k, v in label_mapping.items()}

# Assign distinct colors for each class
class_colors = {
    0: (255, 0, 0),    # Red for main_processor
    1: (0, 255, 0),    # Green for trch
    2: (0, 0, 255),    # Blue for sram
    3: (255, 255, 0),  # Yellow for nor_flash
    4: (0, 255, 255),  # Cyan for fram
    5: (255, 0, 255),  # Magenta for thermal_sensor
    6: (128, 128, 128) # Gray for resistor
}

# Load the model
def get_model(num_classes):
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn(weights=weights)
    
    # Modify the classifier head to match the number of classes (including background)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def load_model(model_path, num_classes, device):
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval()  # Set to evaluation mode
    return model

# Transform to tensor
def transform_image(image):
    transform = transforms.Compose([transforms.ToTensor()])
    return transform(image).unsqueeze(0)

# Function to visualize the detections
def visualize(image, boxes, labels, scores, score_threshold=0.01):
    draw = ImageDraw.Draw(image)
    
    # Load a font with a larger size
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)  # Change the path and size as needed
    except IOError:
        font = ImageFont.load_default()  # Fall back to default font if custom font fails

    for box, label, score in zip(boxes, labels, scores):
        if score > score_threshold:  # Filter out low-confidence boxes
            xmin, ymin, xmax, ymax = box[:4]
            color = class_colors.get(label.item(), (255, 255, 255))  # Default to white if label is unknown
            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
            label_name = label_names.get(label.item(), "Unknown")
            draw.text((xmin, ymin), f'{label_name}: {score:.2f}', fill=color, font=font)
    
    # Save the image instead of displaying it
    output_image_path = "./output_image.jpg"  # Replace with desired output path
    image.save(output_image_path)
    print(f"Image saved to {output_image_path}")


# Inference script
if __name__ == "__main__":
    # File paths
    test_image_path = "./test_image.jpg"
    model_path = "faster_rcnn_resnet50.pth"  

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 7 + 1  # Including background
    model = load_model(model_path, num_classes, device)

    # Load and preprocess the image
    image = Image.open(test_image_path).convert("RGB")
    image_tensor = transform_image(image).to(device)

    # Perform inference
    with torch.no_grad():
        prediction = model(image_tensor)
    
    # Debugging outputs
    print("Prediction:", prediction)

    # Get boxes, labels, and scores
    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu()
    scores = prediction[0]['scores'].cpu().numpy()

    # Visualize and save the results
    visualize(image, boxes, labels, scores)
