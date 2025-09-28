import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from models import UNet, load_model
from evaluater import GlaucomaModelEvaluator, FlexibleDataLoader
from calculationTerms import CalculationUNET


# -----------------------------
# Custom Dataset
# -----------------------------
class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, target_size=(256, 256)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx])

        # Convert mask into 2-channel (OD, OC)
        mask = torch.zeros((2, *self.target_size), dtype=torch.float32)
        mask_np = np.array(mask)
        od = (mask_np == 128).astype(np.float32)
        oc = (mask_np == 0).astype(np.float32)

        mask[0] = torch.tensor(od)
        mask[1] = torch.tensor(oc)

        if self.transform:
            img = self.transform(img)

        return img, mask


# -----------------------------
# Training Function
# -----------------------------
def train_model(
    train_loader, val_loader, model, device,
    epochs=20, lr=1e-4, save_path="finetuned_seg.pth"
):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    calculator = CalculationUNET()

    history = {"train_loss": [], "val_loss": [], "val_dice": []}
    best_val_dice = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        # ------------------ Validation ------------------
        model.eval()
        val_loss = 0.0
        dice_scores = []

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)

                loss = criterion(outputs, masks)
                val_loss += loss.item()

                preds = (outputs >= 0.5).float()
                dice = calculator.compute_dice_coef(preds.cpu(), masks.cpu())
                dice_scores.append(dice)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = sum(dice_scores) / len(dice_scores)

        history["val_loss"].append(avg_val_loss)
        history["val_dice"].append(avg_val_dice)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Dice: {avg_val_dice:.4f}")

        # Save best model
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model with Dice {best_val_dice:.4f}")

    # ------------------ Plot Loss Curves ------------------
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return history


# -----------------------------
# Main
# -----------------------------
def main():
    device = torch.device("cpu")

    # Load pretrained model
    model = UNet()
    pretrained_path = "/home/ankritrisal/Documents/project glaucoma /mainApp/binModel/best_seg.pth"
    model = load_model(model, pretrained_path, device)
    model.train()   # enable training mode

    # Load dataset
    train_loader = FlexibleDataLoader(
        image_dir="/home/ankritrisal/Downloads/NAAMIBPEYE/DATASET /GLAUCOMA/REFUGE2/train/images",
        mask_dir="/home/ankritrisal/Downloads/NAAMIBPEYE/DATASET /GLAUCOMA/REFUGE2/train/mask"
    )
    val_loader = FlexibleDataLoader(
        image_dir="/home/ankritrisal/Downloads/NAAMIBPEYE/DATASET /GLAUCOMA/REFUGE2/val/images",
        mask_dir="/home/ankritrisal/Downloads/NAAMIBPEYE/DATASET /GLAUCOMA/REFUGE2/val/mask"
    )

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    train_dataset = SegmentationDataset(*train_loader.get_data(), transform=transform)
    val_dataset = SegmentationDataset(*val_loader.get_data(), transform=transform)

    train_dl = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Train and fine-tune
    train_model(train_dl, val_dl, model, device, epochs=25, lr=1e-4,
                save_path="finetuned_seg.pth")

    # Reload best and evaluate
    best_model = UNet()
    best_model.load_state_dict(torch.load("finetuned_seg.pth", map_location=device))
    best_model.to(device).eval()

    evaluator = GlaucomaModelEvaluator(best_model, device=device)
    results = evaluator.evaluate_dataset(*val_loader.get_data())
    evaluator.print_summary()


if __name__ == "__main__":
    main()
