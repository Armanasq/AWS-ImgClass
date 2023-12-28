# predict.py

import argparse
import json
import torch
from torchvision import models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# Define command line arguments
parser = argparse.ArgumentParser(description="Predict flower name from an image.")
parser.add_argument('input', type=str, help='Image path')
parser.add_argument('checkpoint', type=str, help='Checkpoint file path')
parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to JSON file mapping categories to real names')
parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

args = parser.parse_args()

# Check for GPU availability
device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

def process_image(image_path):
    # Load the image
    img = Image.open(image_path)
    
    # Resize the images where the shortest side is 256 pixels, keeping the aspect ratio
    img.thumbnail((256, 256))
    
    # Center crop the image
    width, height = img.size
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2
    img = img.crop((left, top, right, bottom))
    
    # Convert to NumPy array
    np_image = np.array(img) / 255.0
    
    # Normalize the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # Transpose the color channel
    np_image = np_image.transpose((2, 0, 1))
    
    return torch.from_numpy(np_image).unsqueeze(0).float()
    
# Define function to load a checkpoint and rebuild the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=device)

    # Initialize the pre-trained model
    model = models.vgg16(pretrained=True) if 'vgg16' == checkpoint['arch'] else None
    if model is None:
        print("Architecture not recognized.")
        return None
    
    # Build the classifier from the checkpoint
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.to(device)
    
    return model

# Load category names
with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

# Load the model
model = load_checkpoint(args.checkpoint)
if model is None:
    raise Exception("Model could not be created. Exiting...")

# Process the image and predict the class
image_path = args.input
image = process_image(image_path).to(device)

# Make prediction
model.eval()
with torch.no_grad():
    output = model(image)
    ps = torch.exp(output)

# Get the top K probabilities and indices
top_p, top_indices = ps.topk(args.top_k, dim=1)
top_p = top_p.tolist()[0]  # Extract values
top_indices = top_indices.tolist()[0]  # Extract values

# Convert indices to classes
idx_to_class = {val: key for key, val in model.class_to_idx.items()}
top_classes = [idx_to_class[index] for index in top_indices]
top_flowers = [cat_to_name[cls] for cls in top_classes]

# Print the results
print("Probabilities:", top_p)
print("Classes:", top_classes)
print("Flower names:", top_flowers)