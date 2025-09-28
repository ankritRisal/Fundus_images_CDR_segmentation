import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
from matplotlib.patches import Circle
from skimage import measure
import cv2

from calculationTerms import CalculationUNET
from dataPreprocessor import preprocess_image


class UnetOutput:
    """
    A class for handling UNET model inference and visualization for optic disc and cup segmentation.
    Enhanced with contour-style visualization and comprehensive CDR analysis.
    """
    
    def __init__(self, device="cuda"):
        """
        Initialize the UnetOutput class.
        
        Args:
            device (str): Device to run inference on. Default is "cuda".
        """
        self.device = device
        self.calculator = CalculationUNET()  # Initialize the calculator instance
    
    def predict_segmentation(self, model, img_path, transform=None):
        """
        Run prediction for single image and compute segmentation masks with vCDR and hCDR.
        
        Args:
            model: Trained segmentation model
            img_path (str): Path to input image
            transform: Optional image transformation
            
        Returns:
            dict: Dictionary containing predictions and computed metrics
        """
        # ---- Preprocess input ----
        img = preprocess_image(img_path, transform)
        sample_img = img.numpy()
        img = img.unsqueeze(0).to(self.device)

        # ---- Forward pass ----
        with torch.no_grad():
            logits = model(img)

            # Predictions (binary masks)
            pred_od = self.calculator.refine_seg((logits[:,0,:,:] >= 0.5).type(torch.int8).cpu())
            pred_oc = self.calculator.refine_seg((logits[:,1,:,:] >= 0.5).type(torch.int8).cpu())

        # Convert to numpy
        pred_od_np = pred_od[0].numpy()
        pred_oc_np = pred_oc[0].numpy()

        # ---- Compute vertical and horizontal diameters ----
        od_vertical_diameter = self.calculator.compute_vertical_diameter(pred_od_np)    # max width
        oc_vertical_diameter = self.calculator.compute_vertical_diameter(pred_oc_np)    # max width  
        od_horizontal_diameter = self.calculator.compute_horizontal_diameter(pred_od_np) # max height
        oc_horizontal_diameter = self.calculator.compute_horizontal_diameter(pred_oc_np) # max height

        # ---- Compute CDRs ----
        vCDR = self.calculator.vertical_cup_to_disc_ratio(pred_od_np, pred_oc_np)
        hCDR = self.calculator.horizontal_cup_to_disc_ratio(pred_od_np, pred_oc_np)

        # ---- Compute centroids and contours for visualization ----
        od_contours = []
        oc_contours = []
        od_centroid = None
        oc_centroid = None
        
        # Find contours for OD
        if np.any(pred_od_np > 0):
            od_coords = np.where(pred_od_np > 0)
            od_centroid = (np.mean(od_coords[1]), np.mean(od_coords[0]))  # (x, y)
            
            # Find contours using OpenCV or skimage
            try:
                contours = measure.find_contours(pred_od_np, 0.5)
                od_contours = contours
            except:
                # Fallback: create approximate contour from coordinates
                od_contours = []
        
        # Find contours for OC
        if np.any(pred_oc_np > 0):
            oc_coords = np.where(pred_oc_np > 0)
            oc_centroid = (np.mean(oc_coords[1]), np.mean(oc_coords[0]))  # (x, y)
            
            try:
                contours = measure.find_contours(pred_oc_np, 0.5)
                oc_contours = contours
            except:
                oc_contours = []

        # ---- Compute bounding boxes for visualization ----
        disc_coords = np.where(pred_od_np > 0)
        cup_coords = np.where(pred_oc_np > 0)
        
        measurements = None
        if len(disc_coords[0]) > 0 and len(cup_coords[0]) > 0:
            # Calculate disc dimensions
            disc_min_row, disc_max_row = disc_coords[0].min(), disc_coords[0].max()
            disc_min_col, disc_max_col = disc_coords[1].min(), disc_coords[1].max()
            
            # Calculate cup dimensions  
            cup_min_row, cup_max_row = cup_coords[0].min(), cup_coords[0].max()
            cup_min_col, cup_max_col = cup_coords[1].min(), cup_coords[1].max()
            
            measurements = {
                'disc_bbox': (disc_min_row, disc_min_col, disc_max_row, disc_max_col),
                'cup_bbox': (cup_min_row, cup_min_col, cup_max_row, cup_max_col),
                'disc_height': disc_max_row - disc_min_row,
                'disc_width': disc_max_col - disc_min_col,
                'cup_height': cup_max_row - cup_min_row,
                'cup_width': cup_max_col - cup_min_col,
                'disc_center_row': (disc_min_row + disc_max_row) / 2,
                'disc_center_col': (disc_min_col + disc_max_col) / 2,
                'cup_center_row': (cup_min_row + cup_max_row) / 2,
                'cup_center_col': (cup_min_col + cup_max_col) / 2
            }

        results = {
            "input_image": sample_img,
            "od_vertical_diameter": od_vertical_diameter,      # ← FIX: Correct names
            "oc_vertical_diameter": oc_vertical_diameter,      # ← FIX: Correct names
            "od_horizontal_diameter": od_horizontal_diameter,  # ← FIX: Correct names  
            "oc_horizontal_diameter": oc_horizontal_diameter,
            "pred_od": pred_od_np,
            "pred_oc": pred_oc_np,
            "pred_vCDR": vCDR,
            "pred_hCDR": hCDR,
            "measurements": measurements,
            "od_contours": od_contours,
            "oc_contours": oc_contours,
            "od_centroid": od_centroid,
            "oc_centroid": oc_centroid
        }

        return results

    def plot_contour_style(self, results, gt_path=None, figsize=(24, 8)):
        """
        Plot results with contour-style visualization using CORRECT diameter calculations
        that match the CDR computation method.
        """
        fig, axes = plt.subplots(1, 4, figsize=figsize)
        
        # Prepare image for display
        if results["input_image"].shape[0] == 3:  # If channels first
            display_image = results["input_image"].transpose(1, 2, 0)
        else:
            display_image = results["input_image"]
        
        # Plot 1: Original Fundus Image
        axes[0].imshow(display_image)
        axes[0].set_title('Original Fundus Image', fontsize=14, fontweight='bold', pad=10)
        axes[0].axis('off')
        
        # Plot 2: Ground Truth Image or Prediction Masks
        gt_vCDR = None
        gt_hCDR = None
        
        if gt_path is not None:
            try:
                # Load and display the actual GT image file
                gt_mask = np.array(Image.open(gt_path).convert('L'))
                gt_od = (gt_mask >= 128).astype(np.uint8)
                gt_oc = (gt_mask == 255).astype(np.uint8)
                
                # Calculate GT CDRs using the same method as predictions
                gt_od_vertical_diameter = self.calculator.compute_vertical_diameter(gt_od.astype(np.float32))
                gt_oc_vertical_diameter = self.calculator.compute_vertical_diameter(gt_oc.astype(np.float32))
                gt_od_horizontal_diameter = self.calculator.compute_horizontal_diameter(gt_od.astype(np.float32))
                gt_oc_horizontal_diameter = self.calculator.compute_horizontal_diameter(gt_oc.astype(np.float32))
                
                # Calculate GT CDRs
                gt_vCDR = gt_oc_vertical_diameter / (gt_od_vertical_diameter + 1e-8)
                gt_hCDR = gt_oc_horizontal_diameter / (gt_od_horizontal_diameter + 1e-8)
                
                # Show the actual GT mask image (grayscale)
                axes[1].imshow(gt_mask, cmap='gray', vmin=0, vmax=255)
                axes[1].set_title('Ground Truth Mask\n(Gray: OD, White: OC)', fontsize=14, fontweight='bold', pad=10)
                axes[1].axis('off')
                
            except Exception as e:
                print(f"Warning: Could not load ground truth from {gt_path}: {e}")
                # Fallback to prediction visualization
                axes[1].imshow(display_image, alpha=0.7)
                axes[1].imshow(results["pred_od"], cmap='Reds', alpha=0.4)
                axes[1].imshow(results["pred_oc"], cmap='Blues', alpha=0.4)
                axes[1].set_title('Predicted Masks\n(Red: OD, Blue: OC)', fontsize=14, fontweight='bold', pad=10)
                axes[1].axis('off')
        else:
            # No GT available - show prediction masks
            axes[1].imshow(display_image, alpha=0.7)
            axes[1].imshow(results["pred_od"], cmap='Reds', alpha=0.4)
            axes[1].imshow(results["pred_oc"], cmap='Blues', alpha=0.4)
            axes[1].set_title('Predicted Masks\n(Red: OD, Blue: OC)', fontsize=14, fontweight='bold', pad=10)
            axes[1].axis('off')
        
        # Get the ACTUAL diameters used in CDR calculation
        pred_od_np = results["pred_od"]
        pred_oc_np = results["pred_oc"]
        
        # Use the SAME method as CDR calculation
        od_vertical_diameter = results["od_vertical_diameter"]
        oc_vertical_diameter = results["oc_vertical_diameter"] 
        od_horizontal_diameter = results["od_horizontal_diameter"]
        oc_horizontal_diameter = results["oc_horizontal_diameter"]
        
        # Plot 3: Vertical CDR Analysis - NOW SHOWS VERTICAL LINES (Height measurement)
        axes[2].imshow(display_image)
        
        # Draw prediction contours
        if results["od_contours"]:
            for contour in results["od_contours"]:
                axes[2].plot(contour[:, 1], contour[:, 0], 'red', linewidth=3, alpha=0.9)
        
        if results["oc_contours"]:
            for contour in results["oc_contours"]:
                axes[2].plot(contour[:, 1], contour[:, 0], 'blue', linewidth=3, alpha=0.9)
        
        # Find the column with maximum height for vertical measurement
        if np.any(pred_od_np > 0):
            od_col_heights = np.sum(pred_od_np > 0, axis=0)
            max_od_col = np.argmax(od_col_heights)
            od_col_indices = np.where(pred_od_np[:, max_od_col] > 0)[0]
            
            if len(od_col_indices) > 0:
                od_top = od_col_indices[0]
                od_bottom = od_col_indices[-1]
                
                # Draw VERTICAL line for vertical diameter (height)
                axes[2].plot([max_od_col, max_od_col], [od_top, od_bottom], 
                            'black', linewidth=4, alpha=0.8)
                
                # Add arrows at the ends
                axes[2].annotate('', xy=(max_od_col, od_top), 
                                xytext=(max_od_col, od_top-10),
                                arrowprops=dict(arrowstyle='->', color='black', lw=2))
                axes[2].annotate('', xy=(max_od_col, od_bottom), 
                                xytext=(max_od_col, od_bottom+10),
                                arrowprops=dict(arrowstyle='->', color='black', lw=2))
                
                axes[2].text(0.95, 0.95, f'OD: {od_vertical_diameter}px', 
                            transform=axes[2].transAxes,
                            fontsize=11, fontweight='bold', ha='right', va='top',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
        
        # Similar for OC vertical measurement
        if np.any(pred_oc_np > 0):
            oc_col_heights = np.sum(pred_oc_np > 0, axis=0)
            max_oc_col = np.argmax(oc_col_heights)
            oc_col_indices = np.where(pred_oc_np[:, max_oc_col] > 0)[0]
            
            if len(oc_col_indices) > 0:
                oc_top = oc_col_indices[0]
                oc_bottom = oc_col_indices[-1]
                
                # Draw VERTICAL line for cup (offset to avoid overlap)
                axes[2].plot([max_oc_col+5, max_oc_col+5], [oc_top, oc_bottom], 
                            'navy', linewidth=4, alpha=0.8)
                
                axes[2].text(0.95, 0.85, f'OC: {oc_vertical_diameter}px', 
                            transform=axes[2].transAxes,
                            fontsize=11, fontweight='bold', ha='right', va='top',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.9))
        
        # Verify CDR calculation
        calculated_vCDR = oc_vertical_diameter / (od_vertical_diameter + 1e-8)
        axes[2].text(0.5, -0.12, f'Predicted vCDR = {calculated_vCDR:.3f}', 
                    transform=axes[2].transAxes, fontsize=12, fontweight='bold',
                    ha='center', va='top',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor='lightgreen', alpha=0.8))
        
        axes[2].set_title('Vertical CDR Analysis\n(Max Height Method)', fontsize=14, fontweight='bold', pad=10)
        axes[2].axis('off')
        
        # Plot 4: Horizontal CDR Analysis - NOW SHOWS HORIZONTAL LINES (Width measurement)
        axes[3].imshow(display_image)
        
        # Draw prediction contours
        if results["od_contours"]:
            for contour in results["od_contours"]:
                axes[3].plot(contour[:, 1], contour[:, 0], 'red', linewidth=3, alpha=0.9)
        
        if results["oc_contours"]:
            for contour in results["oc_contours"]:
                axes[3].plot(contour[:, 1], contour[:, 0], 'blue', linewidth=3, alpha=0.9)
        
        # Find the row with maximum width for horizontal measurement
        if np.any(pred_od_np > 0):
            od_row_widths = np.sum(pred_od_np > 0, axis=1)
            max_od_row = np.argmax(od_row_widths)
            od_row_indices = np.where(pred_od_np[max_od_row, :] > 0)[0]
            
            if len(od_row_indices) > 0:
                od_left = od_row_indices[0]
                od_right = od_row_indices[-1]
                
                # Draw HORIZONTAL line for horizontal diameter (width)
                axes[3].plot([od_left, od_right], [max_od_row, max_od_row], 
                            'black', linewidth=4, alpha=0.8)
                
                # Add arrows
                axes[3].annotate('', xy=(od_left, max_od_row), 
                                xytext=(od_left-10, max_od_row),
                                arrowprops=dict(arrowstyle='->', color='black', lw=2))
                axes[3].annotate('', xy=(od_right, max_od_row), 
                                xytext=(od_right+10, max_od_row),
                                arrowprops=dict(arrowstyle='->', color='black', lw=2))
                
                axes[3].text(0.95, 0.95, f'OD: {od_horizontal_diameter}px', 
                            transform=axes[3].transAxes,
                            fontsize=11, fontweight='bold', ha='right', va='top',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
        
        # Similar for OC horizontal
        if np.any(pred_oc_np > 0):
            oc_row_widths = np.sum(pred_oc_np > 0, axis=1)
            max_oc_row = np.argmax(oc_row_widths)
            oc_row_indices = np.where(pred_oc_np[max_oc_row, :] > 0)[0]
            
            if len(oc_row_indices) > 0:
                oc_left = oc_row_indices[0]
                oc_right = oc_row_indices[-1]
                
                # Draw HORIZONTAL line for cup (offset to avoid overlap)
                axes[3].plot([oc_left, oc_right], [max_oc_row-5, max_oc_row-5], 
                            'navy', linewidth=4, alpha=0.8)
                
                axes[3].text(0.95, 0.85, f'OC: {oc_horizontal_diameter}px', 
                            transform=axes[3].transAxes,
                            fontsize=11, fontweight='bold', ha='right', va='top',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.9))
        
        # Verify hCDR calculation
        calculated_hCDR = oc_horizontal_diameter / (od_horizontal_diameter + 1e-8)
        axes[3].text(0.5, -0.12, f'Predicted hCDR = {calculated_hCDR:.3f}', 
                    transform=axes[3].transAxes, fontsize=12, fontweight='bold',
                    ha='center', va='top',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor='lightyellow', alpha=0.8))
        
        axes[3].set_title('Horizontal CDR Analysis\n(Max Width Method)', fontsize=14, fontweight='bold', pad=10)
        axes[3].axis('off')
        
        # Overall assessment with GT comparison when available
        vCDR = results['pred_vCDR']
        hCDR = results['pred_hCDR']
        max_cdr = max(vCDR, hCDR)
        
        if max_cdr > 0.6:
            risk_level = "HIGH RISK"
            cdr_bg = "mistyrose"
        elif max_cdr > 0.4:
            risk_level = "MODERATE RISK"
            cdr_bg = "peachpuff"
        else:
            risk_level = "LOW RISK"
            cdr_bg = "lightgreen"
        
        # Create assessment text with GT comparison if available
        if gt_vCDR is not None and gt_hCDR is not None:
            # Show both predicted and GT CDRs
            assessment_text = (f'PREDICTED: vCDR = {vCDR:.3f} | hCDR = {hCDR:.3f} | Max CDR = {max_cdr:.3f} | Risk: {risk_level}\n'
                            f'GROUND TRUTH: vCDR = {gt_vCDR:.3f} | hCDR = {gt_hCDR:.3f} | Max GT CDR = {max(gt_vCDR, gt_hCDR):.3f}')
            
            # Calculate errors
            vCDR_error = abs(vCDR - gt_vCDR)
            hCDR_error = abs(hCDR - gt_hCDR)
            assessment_text += f'\nERRORS: vCDR error = {vCDR_error:.3f} | hCDR error = {hCDR_error:.3f}'
            
            text_y_pos = -0.02
        else:
            # Only predicted CDRs
            assessment_text = f'PREDICTED: vCDR = {vCDR:.3f} | hCDR = {hCDR:.3f} | Max CDR = {max_cdr:.3f} | Risk Level: {risk_level}'
            text_y_pos = 0.02
        
        fig.text(0.5, text_y_pos, assessment_text, 
                fontsize=12, fontweight='bold', ha='center', va='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor=cdr_bg, alpha=0.9))
        
        # Adjust layout to accommodate the text
        if gt_vCDR is not None and gt_hCDR is not None:
            plt.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.18, wspace=0.08)
        else:
            plt.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.12, wspace=0.08)
        
        plt.show()

    def plot_results(self, results, gt_path=None, figsize=(18, 6)):
        """
        Original plot method - now redirects to contour style for consistency.
        """
        self.plot_contour_style(results, gt_path, figsize)


    def get_CDRs(self, model, img_path, transform=None):
        """
        Quick method to get both vCDR and hCDR values without plotting.
        
        Args:
            model: Trained segmentation model
            img_path (str): Path to input image
            transform: Optional image transformation
            
        Returns:
            tuple: (vCDR, hCDR) values
        """
        results = self.predict_segmentation(model, img_path, transform)
        return results['pred_vCDR'], results['pred_hCDR']

    def get_vCDR(self, model, img_path, transform=None):
        """
        Quick method to get only the vCDR value without plotting.
        
        Args:
            model: Trained segmentation model
            img_path (str): Path to input image
            transform: Optional image transformation
            
        Returns:
            float: Predicted vCDR value
        """
        results = self.predict_segmentation(model, img_path, transform)
        return results['pred_vCDR']

    def get_hCDR(self, model, img_path, transform=None):
        """
        Quick method to get only the hCDR value without plotting.
        
        Args:
            model: Trained segmentation model
            img_path (str): Path to input image
            transform: Optional image transformation
            
        Returns:
            float: Predicted hCDR value
        """
        results = self.predict_segmentation(model, img_path, transform)
        return results['pred_hCDR']

    def batch_predict(self, model, img_paths, transform=None):
        """
        Run prediction on multiple images.
        Enhanced with horizontal diameter metrics.
        
        Args:
            model: Trained segmentation model
            img_paths (list): List of paths to input images
            transform: Optional image transformation
            
        Returns:
            list: List of results dictionaries for each image
        """
        batch_results = []
        for img_path in img_paths:
            try:
                result = self.predict_segmentation(model, img_path, transform)
                result['img_path'] = img_path  # Add path for reference
                batch_results.append(result)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        return batch_results

    def evaluate_with_gt(self, model, img_path, gt_path, transform=None):
        """
        Run prediction and evaluate against ground truth if available.
        Enhanced with horizontal CDR evaluation and PROPER gt_vCDR/gt_hCDR calculation.
        
        Args:
            model: Trained segmentation model
            img_path (str): Path to input image
            gt_path (str): Path to ground truth mask
            transform: Optional image transformation
            
        Returns:
            dict: Results with additional evaluation metrics
        """
        results = self.predict_segmentation(model, img_path, transform)
        
        try:
            # Load ground truth
            gt_mask = np.array(Image.open(gt_path).convert('L'))
            gt_od = (gt_mask >= 128).astype(np.float32)
            gt_oc = (gt_mask == 255).astype(np.float32)
            
            # Convert predictions to proper format for evaluation
            pred_od_batch = torch.tensor(results['pred_od']).unsqueeze(0)
            pred_oc_batch = torch.tensor(results['pred_oc']).unsqueeze(0)
            gt_od_batch = torch.tensor(gt_od).unsqueeze(0)
            gt_oc_batch = torch.tensor(gt_oc).unsqueeze(0)
            
            # Compute dice scores
            dice_od = self.calculator.compute_dice_coef(pred_od_batch, gt_od_batch)
            dice_oc = self.calculator.compute_dice_coef(pred_oc_batch, gt_oc_batch)
            
            # ============================================================================
            # FIXED: Compute ground truth diameters using the SAME method as predictions
            # ============================================================================
            
            # Ground truth diameters using CONSISTENT naming
            gt_od_vertical_diameter = self.calculator.compute_vertical_diameter(gt_od)      # max width for vCDR
            gt_oc_vertical_diameter = self.calculator.compute_vertical_diameter(gt_oc)      # max width for vCDR
            gt_od_horizontal_diameter = self.calculator.compute_horizontal_diameter(gt_od)  # max height for hCDR
            gt_oc_horizontal_diameter = self.calculator.compute_horizontal_diameter(gt_oc)  # max height for hCDR
            
            # Compute ground truth CDRs manually to ensure consistency
            gt_vCDR_calculated = gt_oc_vertical_diameter / (gt_od_vertical_diameter + 1e-8)
            gt_hCDR_calculated = gt_oc_horizontal_diameter / (gt_od_horizontal_diameter + 1e-8)
            
            # ============================================================================
            # Compute CDR errors using the calculator methods (for validation)
            # ============================================================================
            
            vCDR_error, pred_vCDR, gt_vCDR_from_calc = self.calculator.compute_vCDR_error(
                results['pred_od'].reshape(1, *results['pred_od'].shape),
                results['pred_oc'].reshape(1, *results['pred_oc'].shape),
                gt_od.reshape(1, *gt_od.shape),
                gt_oc.reshape(1, *gt_oc.shape)
            )
            
            hCDR_error, pred_hCDR, gt_hCDR_from_calc = self.calculator.compute_hCDR_error(
                results['pred_od'].reshape(1, *results['pred_od'].shape),
                results['pred_oc'].reshape(1, *results['pred_oc'].shape),
                gt_od.reshape(1, *gt_od.shape),
                gt_oc.reshape(1, *gt_oc.shape)
            )
            
            # ============================================================================
            # Verify consistency (optional debug info)
            # ============================================================================
            print(f"Debug: gt_vCDR calculated manually: {gt_vCDR_calculated:.4f}")
            print(f"Debug: gt_vCDR from calculator: {gt_vCDR_from_calc[0]:.4f}")
            print(f"Debug: gt_hCDR calculated manually: {gt_hCDR_calculated:.4f}")
            print(f"Debug: gt_hCDR from calculator: {gt_hCDR_from_calc[0]:.4f}")
            
            # ============================================================================
            # Add evaluation metrics to results with BOTH calculated values
            # ============================================================================
            results.update({
                'gt_od': gt_od,
                'gt_oc': gt_oc,
                
                # Ground truth diameters (consistent naming)
                'gt_od_vertical_diameter': gt_od_vertical_diameter,      # max width
                'gt_oc_vertical_diameter': gt_oc_vertical_diameter,      # max width
                'gt_od_horizontal_diameter': gt_od_horizontal_diameter,  # max height
                'gt_oc_horizontal_diameter': gt_oc_horizontal_diameter,  # max height
                
                # Legacy naming for backward compatibility (if needed)
                'gt_od_height': gt_od_horizontal_diameter,  # Actually max height
                'gt_oc_height': gt_oc_horizontal_diameter,  # Actually max height  
                'gt_od_width': gt_od_vertical_diameter,     # Actually max width
                'gt_oc_width': gt_oc_vertical_diameter,     # Actually max width
                
                # Dice scores
                'dice_od': dice_od,
                'dice_oc': dice_oc,
                
                # CDR errors
                'vCDR_error': vCDR_error,
                'hCDR_error': hCDR_error,
                
                # Ground truth CDRs - use manually calculated for consistency
                'gt_vCDR': gt_vCDR_calculated,
                'gt_hCDR': gt_hCDR_calculated,
                
                # Also store calculator results for comparison
                'gt_vCDR_from_calc': gt_vCDR_from_calc[0],
                'gt_hCDR_from_calc': gt_hCDR_from_calc[0],
            })
            
            # ============================================================================
            # Print comprehensive evaluation results
            # ============================================================================
            print(f"\n=== Evaluation Results ===")
            print(f"Dice Scores: OD={dice_od:.4f}, OC={dice_oc:.4f}")
            print(f"Predicted CDRs: vCDR={results['pred_vCDR']:.4f}, hCDR={results['pred_hCDR']:.4f}")
            print(f"Ground Truth CDRs: vCDR={gt_vCDR_calculated:.4f}, hCDR={gt_hCDR_calculated:.4f}")
            print(f"CDR Errors: vCDR_error={vCDR_error:.4f}, hCDR_error={hCDR_error:.4f}")
            print(f"Ground Truth Diameters:")
            print(f"  OD: vertical_diameter={gt_od_vertical_diameter}px, horizontal_diameter={gt_od_horizontal_diameter}px")
            print(f"  OC: vertical_diameter={gt_oc_vertical_diameter}px, horizontal_diameter={gt_oc_horizontal_diameter}px")
            print(f"========================\n")
            
        except Exception as e:
            print(f"Warning: Could not evaluate against ground truth: {e}")
        
        return results


    # ============================================================================
    # ALSO UPDATE run_inference to show ground truth CDRs when available
    # ============================================================================

    def run_inference(self, model, img_path, gt_path=None, transform=None, show_plot=True, print_metrics=True, use_enhanced_plot=True):
        """
        Complete pipeline: run inference and optionally plot results.
        Enhanced with ground truth comparison when available.
        
        Args:
            model: Trained segmentation model (should be already loaded)
            img_path (str): Path to input image
            gt_path (str, optional): Path to ground truth mask
            transform: Optional image transformation
            show_plot (bool): Whether to display the plot. Default is True.
            print_metrics (bool): Whether to print metrics to console. Default is True.
            use_enhanced_plot (bool): Whether to use enhanced visualization. Default is True.
            
        Returns:
            dict: Results dictionary containing all predictions and metrics
        """
        # Run prediction with or without ground truth evaluation
        if gt_path is not None:
            results = self.evaluate_with_gt(model, img_path, gt_path, transform)
        else:
            results = self.predict_segmentation(model, img_path, transform)
        
        # Print metrics
        if print_metrics:
            print(f"Prediction Results:")
            print(f"OD Diameters: vertical={results['od_vertical_diameter']:.2f}px, horizontal={results['od_horizontal_diameter']:.2f}px")
            print(f"OC Diameters: vertical={results['oc_vertical_diameter']:.2f}px, horizontal={results['oc_horizontal_diameter']:.2f}px")
            print(f"Predicted CDRs: vCDR={results['pred_vCDR']:.3f}, hCDR={results['pred_hCDR']:.3f}")
            
            # Show ground truth comparison if available
            if 'gt_vCDR' in results and 'gt_hCDR' in results:
                print(f"Ground Truth CDRs: vCDR={results['gt_vCDR']:.3f}, hCDR={results['gt_hCDR']:.3f}")
                print(f"CDR Errors: vCDR_error={results['vCDR_error']:.3f}, hCDR_error={results['hCDR_error']:.3f}")
            
            # Risk assessment
            risk_level = "High" if max(results['pred_vCDR'], results['pred_hCDR']) > 0.6 else \
                        "Moderate" if max(results['pred_vCDR'], results['pred_hCDR']) > 0.4 else "Low"
            print(f"Glaucoma Risk: {risk_level}")
            print("-" * 40)
        
        # Plot results with contour style
        if show_plot:
            self.plot_contour_style(results, gt_path)
        
        return results


# Example usage with contour-style visualization:
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