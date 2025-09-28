import torch
from PIL import Image
from torchvision import transforms

def preprocess_image(img_path, transform=None):
    """Load and preprocess image for model"""
    img = Image.open(img_path).convert("RGB")
    if transform:
        img = transform(img)  # apply same transform used during training
    else:
        # fallback: resize + convert to tensor
        img = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])(img)
    return img


