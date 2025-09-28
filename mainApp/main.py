# In another Python file
import torch
from app import UnetOutput
from models import UNet, load_model
from dataPreprocessor import preprocess_image
from calculationTerms import CalculationUNET

image_path = "/home/ankritrisal/Downloads/NAAMIBPEYE/DATASET /GLAUCOMA/REFUGE2/val/images/V0180.jpg"
gt_Path = "/home/ankritrisal/Downloads/NAAMIBPEYE/DATASET /GLAUCOMA/REFUGE2/val/mask/V0180.png"

model = UNet()
device = torch.device("cpu")

# Initialize
unet_op = UnetOutput(device)
model = load_model(model, "/home/ankritrisal/Documents/project glaucoma /mainApp/binModel/best_seg.pth", device)

# Quick vCDR prediction
# vCDR = unet_op.get_vCDR(model, image_path)
# vCDR, hCDR = unet_op.get_CDRs(model, image_path)

# Full inference with visualization
# results = unet_op.run_inference(model, image_path)
results = unet_op.run_inference(
    model, 
    img_path=image_path,
    gt_path= gt_Path,  # Optional
    show_plot=True
)

# # Method 2: Use contour plotting directly
# results = unet_op.predict_segmentation(model, image_path)
# unet_op.plot_contour_style(results)

# # Method 3: Get CDR values with risk assessment (from model predictions)
# vCDR, hCDR = unet_op.get_CDRs(model, image_path)
# risk_level = "High" if max(vCDR, hCDR) > 0.6 else "Moderate" if max(vCDR, hCDR) > 0.4 else "Low"
# print(f"Model Predictions - vCDR: {vCDR:.3f}, hCDR: {hCDR:.3f}, Risk: {risk_level}")


# print(vCDR, hCDR, results)


"""
# In your main script:
from your_module import UnetOutput

# Initialize the class
unet_op = UnetOutput(device="cuda")

# Load your model
model = load_model(model, "best_seg.pth", device="cuda")

# Method 1: Complete inference with contour-style plotting
results = unet_op.run_inference(
    model, 
    img_path="/path/to/image.jpg",
    show_plot=True
)

# Method 2: Use contour plotting directly
results = unet_op.predict_segmentation(model, "/path/to/image.jpg")
unet_op.plot_contour_style(results)

# Method 3: Get CDR values with clinical interpretation
vCDR, hCDR = unet_op.get_CDRs(model, "/path/to/image.jpg")
max_cdr = max(vCDR, hCDR)
if max_cdr > 0.6:
    risk = "HIGH RISK - Suspect Glaucoma"
elif max_cdr > 0.4:
    risk = "MODERATE RISK - Monitor closely"
else:
    risk = "LOW RISK - Normal"
print(f"Clinical Assessment: vCDR={vCDR:.3f}, hCDR={hCDR:.3f} - {risk}")
"""