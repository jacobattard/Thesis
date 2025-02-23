import torch
import torchvision
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F

# Load Faster R-CNN model (Replace with your trained model if needed)
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

USE_CUSTOM_MODEL = False  # Change to True if using a trained model
CUSTOM_MODEL_PATH = "your_model.pth"

if USE_CUSTOM_MODEL:
    model = torch.load(CUSTOM_MODEL_PATH)
    model.eval()
else:
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.eval()

BALL_CLASS_ID = 0
CONF_THRESH = 0.5

# Load and preprocess the image
image_path = "input_videos/jack_image.png"  # Replace with your image path
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
image_tensor = F.to_tensor(image_rgb).unsqueeze(0)

# Run inference
with torch.no_grad():
    outputs = model(image_tensor)

# Draw detections
for i in range(len(outputs[0]["labels"])):
    label = outputs[0]["labels"][i].item()
    score = outputs[0]["scores"][i].item()
    bbox = outputs[0]["boxes"][i].tolist()  # [x_min, y_min, x_max, y_max]

    if score > CONF_THRESH:
        x_min, y_min, x_max, y_max = map(int, bbox)
        cv2.rectangle(image_rgb, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(image_rgb, f"Conf: {score:.2f}", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show the image with detections using Matplotlib
plt.figure(figsize=(10, 6))
plt.imshow(image_rgb)
plt.axis("off")
plt.show()
