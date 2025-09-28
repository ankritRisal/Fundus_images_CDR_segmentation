import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import transforms
import pandas as pd
from tqdm import tqdm

from models import UNet, load_model
from dataPreprocessor import preprocess_image
from calculationTerms import CalculationUNET


class GlaucomaModelEvaluator:
    """
    Comprehensive evaluation class for UNET segmentation model.
    """
    
    def __init__(self, model, device="cuda", vcdr_threshold=0.6, hcdr_threshold=0.6, output_size=(256, 256)):
        """
        Initialize the evaluator.
        
        Args:
            model: Trained UNET model
            device: Device to run inference on
            vcdr_threshold: Threshold for vertical CDR classification (default: 0.6)
            hcdr_threshold: Threshold for horizontal CDR classification (default: 0.6)
            output_size: Model input/output size
        """
        self.model = model
        self.device = device
        self.vcdr_threshold = vcdr_threshold
        self.hcdr_threshold = hcdr_threshold
        self.output_size = output_size
        self.calculator = CalculationUNET()
        self.results = {}
        
    def load_ground_truth_mask(self, mask_path):
        """
        Load and preprocess ground truth mask following REFUGE dataset format.
        
        Args:
            mask_path: Path to ground truth mask image
            
        Returns:
            tuple: (od_mask, oc_mask) - optic disc and optic cup masks as torch tensors
        """
        # Load mask exactly like in your training dataset loader
        mask = np.array(Image.open(mask_path, mode='r'))
        
        # FIXED: Handle validation dataset format where masks have:
        # White background (255), Grey OD (128), Black OC (0)
        od = (mask == 128).astype(np.float32)  # Optic Disc = pixel value 128 (grey)
        oc = (mask == 0).astype(np.float32)    # Optic Cup = pixel value 0 (black)
        
        # Convert to torch tensors and add channel dimension
        od = torch.from_numpy(od[None, :, :])
        oc = torch.from_numpy(oc[None, :, :])
        
        # Resize to match model output size using nearest neighbor interpolation
        od = transforms.functional.resize(od, self.output_size, interpolation=Image.NEAREST)
        oc = transforms.functional.resize(oc, self.output_size, interpolation=Image.NEAREST)
        
        # Remove channel dimension and convert to numpy for evaluation
        od_mask = od.squeeze().numpy()
        oc_mask = oc.squeeze().numpy()
        
        return od_mask, oc_mask
        
    def predict_single_image(self, img_path):
        """
        Get model predictions for a single image.
        
        Args:
            img_path: Path to input image
            
        Returns:
            tuple: (pred_od, pred_oc) - predicted optic disc and optic cup masks
        """
        img = preprocess_image(img_path)
        img = img.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(img)
            pred_od = (logits[:, 0, :, :] >= 0.5).float().cpu()
            pred_oc = (logits[:, 1, :, :] >= 0.5).float().cpu()
            
            # Refine segmentation to keep only largest connected component
            pred_od = self.calculator.refine_seg(pred_od)
            pred_oc = self.calculator.refine_seg(pred_oc)
            
        return pred_od[0].numpy(), pred_oc[0].numpy()
    
    def compute_iou(self, pred, gt):
        """
        Compute Intersection over Union (IoU) score.
        
        Args:
            pred: Predicted mask
            gt: Ground truth mask
            
        Returns:
            float: IoU score
        """
        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        return intersection / union
    
    def compute_dice_score(self, pred, gt):
        """
        Compute Dice coefficient.
        
        Args:
            pred: Predicted mask
            gt: Ground truth mask
            
        Returns:
            float: Dice score
        """
        pred_flat = pred.flatten()
        gt_flat = gt.flatten()
        intersection = (pred_flat * gt_flat).sum()
        return (2. * intersection) / (pred_flat.sum() + gt_flat.sum() + self.calculator.EPS)
    
    def classify_glaucoma(self, vcdr, hcdr):
        """
        Classify glaucoma based on CDR thresholds.
        
        Args:
            vcdr: Vertical cup-to-disc ratio
            hcdr: Horizontal cup-to-disc ratio
            
        Returns:
            int: 1 for glaucoma, 0 for non-glaucoma
        """
        return 1 if (vcdr > self.vcdr_threshold or hcdr > self.hcdr_threshold) else 0
    
    def debug_single_sample(self, img_path, mask_path, save_debug_plot=True):
        """
        Debug function to visualize and understand your data format.
        
        Args:
            img_path: Path to input image
            mask_path: Path to ground truth mask
            save_debug_plot: Whether to save debug visualization
        """
        print(f"Debugging data format for:")
        print(f"Image: {os.path.basename(img_path)}")
        print(f"Mask: {os.path.basename(mask_path)}")
        print("-" * 50)
        
        # Load original image
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img)
        print(f"Original image shape: {img_array.shape}")
        
        # Load original mask
        mask_original = Image.open(mask_path, mode='r')
        mask_array = np.array(mask_original)
        print(f"Original mask shape: {mask_array.shape}")
        print(f"Unique mask values: {np.unique(mask_array)}")
        
        # Load processed mask using our function
        od_mask, oc_mask = self.load_ground_truth_mask(mask_path)
        print(f"Processed OD mask - Shape: {od_mask.shape}, Sum: {od_mask.sum()}")
        print(f"Processed OC mask - Shape: {oc_mask.shape}, Sum: {oc_mask.sum()}")
        
        # Get model predictions
        pred_od, pred_oc = self.predict_single_image(img_path)
        print(f"Predicted OD mask - Shape: {pred_od.shape}, Sum: {pred_od.sum()}")
        print(f"Predicted OC mask - Shape: {pred_oc.shape}, Sum: {pred_oc.sum()}")
        
        # Calculate CDRs
        gt_vcdr = self.calculator.vertical_cup_to_disc_ratio(od_mask, oc_mask)
        gt_hcdr = self.calculator.horizontal_cup_to_disc_ratio(od_mask, oc_mask)
        pred_vcdr = self.calculator.vertical_cup_to_disc_ratio(pred_od, pred_oc)
        pred_hcdr = self.calculator.horizontal_cup_to_disc_ratio(pred_od, pred_oc)
        
        print(f"Ground Truth - vCDR: {gt_vcdr:.4f}, hCDR: {gt_hcdr:.4f}")
        print(f"Predicted - vCDR: {pred_vcdr:.4f}, hCDR: {pred_hcdr:.4f}")
        
        # Classification
        gt_glaucoma = self.classify_glaucoma(gt_vcdr, gt_hcdr)
        pred_glaucoma = self.classify_glaucoma(pred_vcdr, pred_hcdr)
        print(f"Ground Truth Classification: {'Glaucoma' if gt_glaucoma else 'Non-Glaucoma'}")
        print(f"Predicted Classification: {'Glaucoma' if pred_glaucoma else 'Non-Glaucoma'}")
        
        if save_debug_plot:
            # Create visualization
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            
            # Row 1: Original data and ground truth
            axes[0, 0].imshow(img_array)
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            # FIXED: Show original mask in grayscale to see true values
            axes[0, 1].imshow(mask_array, cmap='gray', vmin=0, vmax=255)
            axes[0, 1].set_title(f'Original Mask\nValues: {np.unique(mask_array)}\nWhite=BG(255), Gray=OD(128), Black=OC(0)')
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(od_mask, cmap='Reds', alpha=0.8)
            axes[0, 2].set_title(f'GT Optic Disc\nPixels: {int(od_mask.sum())}')
            axes[0, 2].axis('off')
            
            axes[0, 3].imshow(oc_mask, cmap='Blues', alpha=0.8)
            axes[0, 3].set_title(f'GT Optic Cup\nPixels: {int(oc_mask.sum())}')
            axes[0, 3].axis('off')
            
            # Row 2: Predictions and overlays
            axes[1, 0].imshow(pred_od, cmap='Reds', alpha=0.8)
            axes[1, 0].set_title(f'Pred Optic Disc\nPixels: {int(pred_od.sum())}')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(pred_oc, cmap='Blues', alpha=0.8)
            axes[1, 1].set_title(f'Pred Optic Cup\nPixels: {int(pred_oc.sum())}')
            axes[1, 1].axis('off')
            
            # Overlay GT
            overlay_gt = np.zeros((*od_mask.shape, 3))
            overlay_gt[:, :, 0] = od_mask  # Red for OD
            overlay_gt[:, :, 2] = oc_mask  # Blue for OC
            axes[1, 2].imshow(overlay_gt)
            axes[1, 2].set_title(f'GT Overlay\nvCDR: {gt_vcdr:.3f}, hCDR: {gt_hcdr:.3f}')
            axes[1, 2].axis('off')
            
            # Overlay Predictions
            overlay_pred = np.zeros((*pred_od.shape, 3))
            overlay_pred[:, :, 0] = pred_od  # Red for OD
            overlay_pred[:, :, 2] = pred_oc  # Blue for OC
            axes[1, 3].imshow(overlay_pred)
            axes[1, 3].set_title(f'Pred Overlay\nvCDR: {pred_vcdr:.3f}, hCDR: {pred_hcdr:.3f}')
            axes[1, 3].axis('off')
            
            plt.tight_layout()
            debug_filename = f'debug_{os.path.basename(img_path)[:-4]}.png'
            plt.savefig(debug_filename, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"Debug plot saved as: {debug_filename}")
            
        return {
            'gt_od_pixels': int(od_mask.sum()),
            'gt_oc_pixels': int(oc_mask.sum()),
            'pred_od_pixels': int(pred_od.sum()),
            'pred_oc_pixels': int(pred_oc.sum()),
            'gt_vcdr': gt_vcdr,
            'gt_hcdr': gt_hcdr,
            'pred_vcdr': pred_vcdr,
            'pred_hcdr': pred_hcdr,
            'gt_glaucoma': gt_glaucoma,
            'pred_glaucoma': pred_glaucoma
        }
    
    def evaluate_single_sample(self, img_path, mask_path):
        """
        Evaluate a single sample.
        
        Args:
            img_path: Path to input image
            mask_path: Path to ground truth mask
            
        Returns:
            dict: Evaluation metrics for this sample
        """
        # Get predictions
        pred_od, pred_oc = self.predict_single_image(img_path)
        
        # Load ground truth
        gt_od, gt_oc = self.load_ground_truth_mask(mask_path)
        
        # Compute segmentation metrics
        od_iou = self.compute_iou(pred_od, gt_od)
        oc_iou = self.compute_iou(pred_oc, gt_oc)
        od_dice = self.compute_dice_score(pred_od, gt_od)
        oc_dice = self.compute_dice_score(pred_oc, gt_oc)
        
        # Compute CDRs
        pred_vcdr = self.calculator.vertical_cup_to_disc_ratio(pred_od, pred_oc)
        pred_hcdr = self.calculator.horizontal_cup_to_disc_ratio(pred_od, pred_oc)
        gt_vcdr = self.calculator.vertical_cup_to_disc_ratio(gt_od, gt_oc)
        gt_hcdr = self.calculator.horizontal_cup_to_disc_ratio(gt_od, gt_oc)
        
        # CDR errors
        vcdr_error = abs(pred_vcdr - gt_vcdr)
        hcdr_error = abs(pred_hcdr - gt_hcdr)
        
        # Classification
        pred_glaucoma = self.classify_glaucoma(pred_vcdr, pred_hcdr)
        gt_glaucoma = self.classify_glaucoma(gt_vcdr, gt_hcdr)
        
        return {
            'img_path': img_path,
            'od_iou': od_iou,
            'oc_iou': oc_iou,
            'od_dice': od_dice,
            'oc_dice': oc_dice,
            'pred_vcdr': pred_vcdr,
            'pred_hcdr': pred_hcdr,
            'gt_vcdr': gt_vcdr,
            'gt_hcdr': gt_hcdr,
            'vcdr_error': vcdr_error,
            'hcdr_error': hcdr_error,
            'pred_glaucoma': pred_glaucoma,
            'gt_glaucoma': gt_glaucoma
        }
    
    def evaluate_dataset(self, image_paths, mask_paths):
        """
        Evaluate the model on a dataset.
        
        Args:
            image_paths: List of image paths
            mask_paths: List of corresponding mask paths
            
        Returns:
            dict: Comprehensive evaluation results
        """
        if len(image_paths) != len(mask_paths):
            raise ValueError("Number of images and masks must match")
        
        sample_results = []
        
        print("Evaluating dataset...")
        for img_path, mask_path in tqdm(zip(image_paths, mask_paths), total=len(image_paths)):
            try:
                result = self.evaluate_single_sample(img_path, mask_path)
                sample_results.append(result)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        # Aggregate results
        self.results = self._aggregate_results(sample_results)
        return self.results
    
    def _aggregate_results(self, sample_results):
        """Aggregate individual sample results into overall metrics."""
        df = pd.DataFrame(sample_results)
        
        # Segmentation metrics
        segmentation_metrics = {
            'mean_od_iou': df['od_iou'].mean(),
            'std_od_iou': df['od_iou'].std(),
            'mean_oc_iou': df['oc_iou'].mean(),
            'std_oc_iou': df['oc_iou'].std(),
            'mean_od_dice': df['od_dice'].mean(),
            'std_od_dice': df['od_dice'].std(),
            'mean_oc_dice': df['oc_dice'].mean(),
            'std_oc_dice': df['oc_dice'].std(),
        }
        
        # CDR metrics
        cdr_metrics = {
            'mean_vcdr_error': df['vcdr_error'].mean(),
            'std_vcdr_error': df['vcdr_error'].std(),
            'mean_hcdr_error': df['hcdr_error'].mean(),
            'std_hcdr_error': df['hcdr_error'].std(),
            'mean_pred_vcdr': df['pred_vcdr'].mean(),
            'mean_gt_vcdr': df['gt_vcdr'].mean(),
            'mean_pred_hcdr': df['pred_hcdr'].mean(),
            'mean_gt_hcdr': df['gt_hcdr'].mean(),
        }
        
        # Classification metrics
        y_true = df['gt_glaucoma'].values
        y_pred = df['pred_glaucoma'].values
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Handle case where we might not have both classes
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        elif len(np.unique(y_true)) == 1 and len(np.unique(y_pred)) == 1:
            if y_true[0] == 0 and y_pred[0] == 0:
                tn, fp, fn, tp = len(y_true), 0, 0, 0
            else:
                tn, fp, fn, tp = 0, 0, 0, len(y_true)
        else:
            print("Warning: Unusual confusion matrix shape, setting default values")
            tn, fp, fn, tp = 0, 0, 0, 0
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if len(y_true) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        classification_metrics = {
            'confusion_matrix': cm,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'accuracy': accuracy,
            'precision': precision,
            'f1_score': f1_score,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }
        
        return {
            'sample_results': df,
            'segmentation_metrics': segmentation_metrics,
            'cdr_metrics': cdr_metrics,
            'classification_metrics': classification_metrics,
            'vcdr_threshold': self.vcdr_threshold,
            'hcdr_threshold': self.hcdr_threshold,
            'total_samples': len(sample_results)
        }
    
    def plot_confusion_matrix(self, save_path=None):
        """Plot confusion matrix."""
        if not self.results:
            raise ValueError("No results available. Run evaluation first.")
        
        cm = self.results['classification_metrics']['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Non-Glaucoma', 'Glaucoma'],
                    yticklabels=['Non-Glaucoma', 'Glaucoma'])
        plt.title('Confusion Matrix for Glaucoma Classification')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Add metrics text
        cls_metrics = self.results['classification_metrics']
        metrics_text = f"Accuracy: {cls_metrics['accuracy']:.3f}\n"
        metrics_text += f"Sensitivity: {cls_metrics['sensitivity']:.3f}\n"
        metrics_text += f"Specificity: {cls_metrics['specificity']:.3f}"
        plt.text(2.1, 0.5, metrics_text, fontsize=12, verticalalignment='center')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_cdr_comparison(self, save_path=None):
        """Plot CDR comparison between predictions and ground truth."""
        if not self.results:
            raise ValueError("No results available. Run evaluation first.")
        
        df = self.results['sample_results']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # vCDR comparison
        ax1.scatter(df['gt_vcdr'], df['pred_vcdr'], alpha=0.7, s=50)
        ax1.plot([0, 1], [0, 1], 'r--', label='Perfect Prediction')
        ax1.axhline(y=self.vcdr_threshold, color='orange', linestyle='--', label=f'Threshold ({self.vcdr_threshold})')
        ax1.axvline(x=self.vcdr_threshold, color='orange', linestyle='--')
        ax1.set_xlabel('Ground Truth vCDR')
        ax1.set_ylabel('Predicted vCDR')
        ax1.set_title('Vertical CDR: Prediction vs Ground Truth')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # hCDR comparison
        ax2.scatter(df['gt_hcdr'], df['pred_hcdr'], alpha=0.7, s=50)
        ax2.plot([0, 1], [0, 1], 'r--', label='Perfect Prediction')
        ax2.axhline(y=self.hcdr_threshold, color='orange', linestyle='--', label=f'Threshold ({self.hcdr_threshold})')
        ax2.axvline(x=self.hcdr_threshold, color='orange', linestyle='--')
        ax2.set_xlabel('Ground Truth hCDR')
        ax2.set_ylabel('Predicted hCDR')
        ax2.set_title('Horizontal CDR: Prediction vs Ground Truth')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_segmentation_metrics(self, save_path=None):
        """Plot segmentation metrics distribution."""
        if not self.results:
            raise ValueError("No results available. Run evaluation first.")
        
        df = self.results['sample_results']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # IoU distributions
        axes[0, 0].hist(df['od_iou'], bins=20, alpha=0.7, label='Optic Disc IoU', color='red')
        axes[0, 0].hist(df['oc_iou'], bins=20, alpha=0.7, label='Optic Cup IoU', color='blue')
        axes[0, 0].set_xlabel('IoU Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('IoU Score Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Dice distributions
        axes[0, 1].hist(df['od_dice'], bins=20, alpha=0.7, label='Optic Disc Dice', color='red')
        axes[0, 1].hist(df['oc_dice'], bins=20, alpha=0.7, label='Optic Cup Dice', color='blue')
        axes[0, 1].set_xlabel('Dice Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Dice Score Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # CDR error distributions
        axes[1, 0].hist(df['vcdr_error'], bins=20, alpha=0.7, label='vCDR Error', color='green')
        axes[1, 0].hist(df['hcdr_error'], bins=20, alpha=0.7, label='hCDR Error', color='purple')
        axes[1, 0].set_xlabel('CDR Error')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('CDR Error Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Box plot of all metrics
        metrics_data = [df['od_iou'], df['oc_iou'], df['od_dice'], df['oc_dice']]
        box_plot = axes[1, 1].boxplot(metrics_data, labels=['OD IoU', 'OC IoU', 'OD Dice', 'OC Dice'], patch_artist=True)
        colors = ['red', 'blue', 'red', 'blue']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Segmentation Metrics Box Plot')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_summary(self):
        """Print comprehensive evaluation summary."""
        if not self.results:
            raise ValueError("No results available. Run evaluation first.")
        
        seg_metrics = self.results['segmentation_metrics']
        cdr_metrics = self.results['cdr_metrics']
        cls_metrics = self.results['classification_metrics']
        
        print("="*80)
        print("GLAUCOMA MODEL EVALUATION SUMMARY")
        print("="*80)
        print(f"Total samples evaluated: {self.results['total_samples']}")
        
        print(f"\nSEGMENTATION METRICS:")
        print(f"Optic Disc  - IoU: {seg_metrics['mean_od_iou']:.4f} ± {seg_metrics['std_od_iou']:.4f}")
        print(f"Optic Disc  - Dice: {seg_metrics['mean_od_dice']:.4f} ± {seg_metrics['std_od_dice']:.4f}")
        print(f"Optic Cup   - IoU: {seg_metrics['mean_oc_iou']:.4f} ± {seg_metrics['std_oc_iou']:.4f}")
        print(f"Optic Cup   - Dice: {seg_metrics['mean_oc_dice']:.4f} ± {seg_metrics['std_oc_dice']:.4f}")
        
        print(f"\nCDR METRICS:")
        print(f"vCDR Error: {cdr_metrics['mean_vcdr_error']:.4f} ± {cdr_metrics['std_vcdr_error']:.4f}")
        print(f"hCDR Error: {cdr_metrics['mean_hcdr_error']:.4f} ± {cdr_metrics['std_hcdr_error']:.4f}")
        print(f"Mean Predicted vCDR: {cdr_metrics['mean_pred_vcdr']:.4f}")
        print(f"Mean Ground Truth vCDR: {cdr_metrics['mean_gt_vcdr']:.4f}")
        print(f"Mean Predicted hCDR: {cdr_metrics['mean_pred_hcdr']:.4f}")
        print(f"Mean Ground Truth hCDR: {cdr_metrics['mean_gt_hcdr']:.4f}")
        
        print(f"\nCLASSIFICATION METRICS (Thresholds: vCDR>{self.vcdr_threshold}, hCDR>{self.hcdr_threshold}):")
        print(f"Accuracy: {cls_metrics['accuracy']:.4f}")
        print(f"Sensitivity (Recall): {cls_metrics['sensitivity']:.4f}")
        print(f"Specificity: {cls_metrics['specificity']:.4f}")
        print(f"Precision: {cls_metrics['precision']:.4f}")
        print(f"F1-Score: {cls_metrics['f1_score']:.4f}")
        
        print(f"\nCONFUSION MATRIX:")
        print(f"True Positives (Glaucoma correctly identified): {cls_metrics['true_positives']}")
        print(f"True Negatives (Non-glaucoma correctly identified): {cls_metrics['true_negatives']}")
        print(f"False Positives (Non-glaucoma misclassified as glaucoma): {cls_metrics['false_positives']}")
        print(f"False Negatives (Glaucoma misclassified as non-glaucoma): {cls_metrics['false_negatives']}")
        
        print("="*80)
    
    def save_results(self, save_dir):
        """Save evaluation results to files."""
        if not self.results:
            raise ValueError("No results available. Run evaluation first.")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save detailed results
        self.results['sample_results'].to_csv(
            os.path.join(save_dir, 'detailed_results.csv'), index=False)
        
        # Save summary metrics
        summary = {**self.results['segmentation_metrics'], 
                  **self.results['cdr_metrics'], 
                  **{k: v for k, v in self.results['classification_metrics'].items() 
                     if k != 'confusion_matrix'}}
        
        with open(os.path.join(save_dir, 'summary_metrics.txt'), 'w') as f:
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
        
        # Save confusion matrix
        np.savetxt(os.path.join(save_dir, 'confusion_matrix.csv'), 
                   self.results['classification_metrics']['confusion_matrix'], 
                   delimiter=',', fmt='%d')
        
        print(f"Results saved to {save_dir}")


class FlexibleDataLoader:
    """
    Flexible dataloader that matches images and masks based on your dataset structure.
    Handles the specific format where images are in one folder and masks in another.
    """
    
    def __init__(self, image_dir, mask_dir, image_extensions=None, mask_extensions=None):
        """
        Initialize the dataloader.
        
        Args:
            image_dir: Directory containing input images
            mask_dir: Directory containing ground truth masks
            image_extensions: List of supported image extensions
            mask_extensions: List of supported mask extensions
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        
        # Default supported extensions
        if image_extensions is None:
            self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        else:
            self.image_extensions = [ext.lower() for ext in image_extensions]
            
        if mask_extensions is None:
            self.mask_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
        else:
            self.mask_extensions = [ext.lower() for ext in mask_extensions]
        
        # Find and match files
        self.image_paths, self.mask_paths = self._match_files()
        
        print(f"Found {len(self.image_paths)} matched image-mask pairs")
    
    def _get_files_with_extensions(self, directory, extensions):
        """Get all files in directory with specified extensions."""
        files = []
        if not os.path.exists(directory):
            print(f"Warning: Directory {directory} does not exist")
            return files
        
        for filename in os.listdir(directory):
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in extensions and not filename.startswith('.'):
                files.append(os.path.join(directory, filename))
        
        return sorted(files)
    
    def _match_files(self):
        """Match images and masks based on base filenames."""
        # Get all image and mask files
        image_files = self._get_files_with_extensions(self.image_dir, self.image_extensions)
        mask_files = self._get_files_with_extensions(self.mask_dir, self.mask_extensions)
        
        # Create dictionaries mapping base names to full paths
        image_dict = {}
        mask_dict = {}
        
        for img_path in image_files:
            # Handle the case where mask might have different extension
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            image_dict[base_name] = img_path
        
        for mask_path in mask_files:
            base_name = os.path.splitext(os.path.basename(mask_path))[0]
            mask_dict[base_name] = mask_path
        
        # Find matching pairs
        matched_images = []
        matched_masks = []
        
        unmatched_images = []
        unmatched_masks = []
        
        for base_name in image_dict:
            if base_name in mask_dict:
                matched_images.append(image_dict[base_name])
                matched_masks.append(mask_dict[base_name])
            else:
                unmatched_images.append(base_name)
        
        # Find unmatched masks
        for base_name in mask_dict:
            if base_name not in image_dict:
                unmatched_masks.append(base_name)
        
        # Print warnings for unmatched files
        if unmatched_images:
            print(f"Warning: {len(unmatched_images)} images without matching masks:")
            for name in unmatched_images[:5]:
                print(f"  - {name}")
            if len(unmatched_images) > 5:
                print(f"  ... and {len(unmatched_images) - 5} more")
        
        if unmatched_masks:
            print(f"Warning: {len(unmatched_masks)} masks without matching images:")
            for name in unmatched_masks[:5]:
                print(f"  - {name}")
            if len(unmatched_masks) > 5:
                print(f"  ... and {len(unmatched_masks) - 5} more")
        
        return matched_images, matched_masks
    
    def get_data(self):
        """
        Get matched image paths and mask paths.
        
        Returns:
            tuple: (image_paths, mask_paths)
        """
        return self.image_paths, self.mask_paths
    
    def print_sample_info(self):
        """Print information about the loaded dataset."""
        print("\n" + "="*50)
        print("DATASET INFORMATION")
        print("="*50)
        print(f"Total samples: {len(self.image_paths)}")
        print(f"Image directory: {self.image_dir}")
        print(f"Mask directory: {self.mask_dir}")
        print(f"Image extensions: {', '.join(self.image_extensions)}")
        print(f"Mask extensions: {', '.join(self.mask_extensions)}")
        print("="*50)


def main():
    """
    Example usage of the GlaucomaModelEvaluator
    """
    
    # Model configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load your trained model
    model = UNet()
    model_path = "/home/ankritrisal/Documents/project glaucoma /mainApp/binModel/best_seg.pth"  # Update this path
    model = load_model(model, model_path, device)
    
    # Initialize evaluator with CDR thresholds
    evaluator = GlaucomaModelEvaluator(
        model=model,
        device=device,
        vcdr_threshold=0.6,  # Clinical threshold for vertical CDR
        hcdr_threshold=0.6,  # Clinical threshold for horizontal CDR
        output_size=(256, 256)  # Match your model's input/output size
    )
    
    # Initialize dataloader
    dataloader = FlexibleDataLoader(
        image_dir="/home/ankritrisal/Downloads/NAAMIBPEYE/DATASET /GLAUCOMA/REFUGE2/val/images",  # Update this path
        mask_dir="/home/ankritrisal/Downloads/NAAMIBPEYE/DATASET /GLAUCOMA/REFUGE2/val/mask",    # Update this path
        image_extensions=['.jpg', '.jpeg', '.png'],
        mask_extensions=['.png']
    )
    
    # Print dataset information
    dataloader.print_sample_info()
    
    # Get matched data
    image_paths, mask_paths = dataloader.get_data()
    
    if len(image_paths) == 0:
        print("No matching image-mask pairs found. Please check your paths.")
        return
    
    # Debug first few samples to verify data loading
    print("\nDebugging first few samples...")
    for i in range(min(3, len(image_paths))):
        print(f"\nDebugging sample {i+1}:")
        debug_result = evaluator.debug_single_sample(
            image_paths[i], mask_paths[i], save_debug_plot=True
        )
        
        # Check if masks are being loaded correctly
        if debug_result['gt_od_pixels'] == 0:
            print("WARNING: Ground truth OD mask is empty!")
        if debug_result['gt_oc_pixels'] == 0:
            print("WARNING: Ground truth OC mask is empty!")
        if debug_result['pred_od_pixels'] == 0:
            print("WARNING: Predicted OD mask is empty!")
        if debug_result['pred_oc_pixels'] == 0:
            print("WARNING: Predicted OC mask is empty!")
    
    # Ask user confirmation
    response = input("\nDo you want to proceed with full evaluation? (y/n): ")
    if response.lower() != 'y':
        print("Evaluation cancelled.")
        return
    
    # Run full evaluation
    print("Running full evaluation...")
    results = evaluator.evaluate_dataset(image_paths, mask_paths)
    
    # Print summary
    evaluator.print_summary()
    
    # Generate plots
    evaluator.plot_confusion_matrix(save_path="confusion_matrix.png")
    evaluator.plot_cdr_comparison(save_path="cdr_comparison.png")
    evaluator.plot_segmentation_metrics(save_path="segmentation_metrics.png")
    
    # Save results
    evaluator.save_results("evaluation_results")
    
    print("\nEvaluation complete! Check the generated plots and saved results.")


# # Example usage for testing specific functionality
# def test_single_sample():
#     """
#     Test function to debug a single sample
#     """
#     # Update these paths to your actual files
#     img_path = "/path/to/sample_image.jpg"
#     mask_path = "/path/to/sample_mask.png"
#     model_path = "/path/to/your/model.pth"
    
#     # Load model
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = UNet()
#     model = load_model(model, model_path, device)
    
#     # Initialize evaluator
#     evaluator = GlaucomaModelEvaluator(model=model, device=device)
    
#     # Debug single sample
#     debug_result = evaluator.debug_single_sample(img_path, mask_path)
#     print("Debug result:", debug_result)


if __name__ == "__main__":
    main()
    # Uncomment the line below to test single sample debugging
    # test_single_sample()


"""
Using device: cpu
✅ Model loaded successfully from /home/ankritrisal/Documents/project glaucoma /mainApp/binModel/best_seg.pth
Found 400 matched image-mask pairs

==================================================
DATASET INFORMATION
==================================================
Total samples: 400
Image directory: /home/ankritrisal/Downloads/NAAMIBPEYE/DATASET /GLAUCOMA/REFUGE2/val/images
Mask directory: /home/ankritrisal/Downloads/NAAMIBPEYE/DATASET /GLAUCOMA/REFUGE2/val/mask
Image extensions: .jpg, .jpeg, .png
Mask extensions: .png
==================================================

Debugging first few samples...

Debugging sample 1:
Debugging data format for:
Image: V0001.jpg
Mask: V0001.png
--------------------------------------------------
Original image shape: (1940, 1940, 3)
Original mask shape: (1940, 1940)
Unique mask values: [  0 128 255]
Processed OD mask - Shape: (256, 256), Sum: 654.0
Processed OC mask - Shape: (256, 256), Sum: 291.0
Predicted OD mask - Shape: (256, 256), Sum: 522.0
Predicted OC mask - Shape: (256, 256), Sum: 230.0
Ground Truth - vCDR: 0.6774, hCDR: 0.6207
Predicted - vCDR: 0.6800, hCDR: 0.6296
Ground Truth Classification: Glaucoma
Predicted Classification: Glaucoma
Debug plot saved as: debug_V0001.png

Debugging sample 2:
Debugging data format for:
Image: V0002.jpg
Mask: V0002.png
--------------------------------------------------
Original image shape: (1940, 1940, 3)
Original mask shape: (1940, 1940)
Unique mask values: [  0 128 255]
Processed OD mask - Shape: (256, 256), Sum: 523.0
Processed OC mask - Shape: (256, 256), Sum: 435.0
Predicted OD mask - Shape: (256, 256), Sum: 403.0
Predicted OC mask - Shape: (256, 256), Sum: 464.0
Ground Truth - vCDR: 1.0588, hCDR: 0.9143
Predicted - vCDR: 1.0526, hCDR: 0.9062
Ground Truth Classification: Glaucoma
Predicted Classification: Glaucoma
Debug plot saved as: debug_V0002.png

Debugging sample 3:
Debugging data format for:
Image: V0003.jpg
Mask: V0003.png
--------------------------------------------------
Original image shape: (1940, 1940, 3)
Original mask shape: (1940, 1940)
Unique mask values: [  0 128 255]
Processed OD mask - Shape: (256, 256), Sum: 582.0
Processed OC mask - Shape: (256, 256), Sum: 289.0
Predicted OD mask - Shape: (256, 256), Sum: 663.0
Predicted OC mask - Shape: (256, 256), Sum: 226.0
Ground Truth - vCDR: 0.6818, hCDR: 0.7742
Predicted - vCDR: 0.5517, hCDR: 0.5806
Ground Truth Classification: Glaucoma
Predicted Classification: Non-Glaucoma
Debug plot saved as: debug_V0003.png

Do you want to proceed with full evaluation? (y/n): y
Running full evaluation...
Evaluating dataset...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 400/400 [03:14<00:00,  2.06it/s]
================================================================================
GLAUCOMA MODEL EVALUATION SUMMARY
================================================================================
Total samples evaluated: 400

SEGMENTATION METRICS:
Optic Disc  - IoU: 0.8002 ± 0.1028
Optic Disc  - Dice: 0.8845 ± 0.0812
Optic Cup   - IoU: 0.7163 ± 0.1514
Optic Cup   - Dice: 0.8232 ± 0.1335

CDR METRICS:
vCDR Error: 0.1336 ± 0.5894
hCDR Error: 0.1280 ± 0.6274
Mean Predicted vCDR: 0.6200
Mean Ground Truth vCDR: 0.5564
Mean Predicted hCDR: 0.5842
Mean Ground Truth hCDR: 0.5249

CLASSIFICATION METRICS (Thresholds: vCDR>0.6, hCDR>0.6):
Accuracy: 0.8500
Sensitivity (Recall): 0.8209
Specificity: 0.8647
Precision: 0.7534
F1-Score: 0.7857

CONFUSION MATRIX:
True Positives (Glaucoma correctly identified): 110
True Negatives (Non-glaucoma correctly identified): 230
False Positives (Non-glaucoma misclassified as glaucoma): 36
False Negatives (Glaucoma misclassified as non-glaucoma): 24
================================================================================

"""