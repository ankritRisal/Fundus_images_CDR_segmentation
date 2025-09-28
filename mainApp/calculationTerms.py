import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from scipy.ndimage import label


class CalculationUNET:
    """
    A class for computing various metrics related to UNET segmentation tasks,
    particularly for optic disc and optic cup segmentation in medical imaging.
    """
    
    def __init__(self, eps=1e-7):
        """
        Initialize the CalculationUNET class.
        
        Args:
            eps (float): Small epsilon value to prevent division by zero. Default is 1e-7.
        """
        self.EPS = eps
    
    def compute_dice_coef(self, input, target):
        """
        Compute dice score metric for batch of samples.
        
        Args:
            input: Predicted segmentation tensor with shape (batch_size, height, width)
            target: Ground truth segmentation tensor with shape (batch_size, height, width)
            
        Returns:
            float: Average dice coefficient across the batch
        """
        batch_size = input.shape[0]
        return sum([self.dice_coef_sample(input[k,:,:], target[k,:,:]) for k in range(batch_size)]) / batch_size
    
    def dice_coef_sample(self, input, target):
        """
        Compute dice coefficient for a single sample.
        
        Args:
            input: Predicted segmentation for single sample
            target: Ground truth segmentation for single sample
            
        Returns:
            float: Dice coefficient for the sample
        """
        iflat = input.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        return (2. * intersection) / (iflat.sum() + tflat.sum())
    
    def compute_vertical_diameter(self, binary_segmentation: np.ndarray):
        """
        Get the vertical diameter from a binary segmentation.
        The vertical diameter is defined as the maximum width across all rows
        (maximum number of nonzero pixels in any single row).
        
        This is used for vCDR calculation - measures the widest horizontal span.
        
        Args:
            binary_segmentation: numpy array (H,W) with 0/1 values

        Returns:
            int: vertical diameter (maximum width in pixels)
        """
        if binary_segmentation.ndim != 2:
            raise ValueError("Input mask must be 2D (H, W)")

        # Count number of nonzero pixels per row (vertical width for each row)
        row_widths = np.sum(binary_segmentation > 0, axis=0)

        # Take maximum width across all rows
        diameter = int(np.max(row_widths)) if len(row_widths) > 0 else 0

        return diameter
    
    def compute_horizontal_diameter(self, binary_segmentation: np.ndarray):
        """
        Get the horizontal diameter from a binary segmentation.
        The horizontal diameter is defined as the maximum height across all columns
        (maximum number of nonzero pixels in any single column).
        
        This is used for hCDR calculation - measures the tallest vertical span.
        
        Args:
            binary_segmentation: numpy array (H,W) with 0/1 values

        Returns:
            int: horizontal diameter (maximum height in pixels)
        """
        if binary_segmentation.ndim != 2:
            raise ValueError("Input mask must be 2D (H, W)")

        # Count number of nonzero pixels per column (horizontal height for each column)
        column_heights = np.sum(binary_segmentation > 0, axis=1)

        # Take maximum height across all columns
        diameter = int(np.max(column_heights)) if len(column_heights) > 0 else 0

        return diameter


    
    def vertical_cup_to_disc_ratio(self, od, oc):
        """
        Compute the vertical cup-to-disc ratio (vCDR) from segmentation masks.
        
        The vCDR is calculated using the maximum width method:
        - Find the row with maximum width for both OD and OC
        - Compute ratio of cup width to disc width
        
        Args:
            od: Optic disc segmentation (numpy array, H x W)
            oc: Optic cup segmentation (numpy array, H x W)
            
        Returns:
            float: Vertical cup-to-disc ratio
        """
        # Compute the maximum width (horizontal span) for cup and disc
        cup_diameter = self.compute_vertical_diameter(oc)
        disc_diameter = self.compute_vertical_diameter(od)
        
        # Return ratio with epsilon to prevent division by zero
        return cup_diameter / (disc_diameter + self.EPS)
    
    def horizontal_cup_to_disc_ratio(self, od, oc):
        """
        Compute the horizontal cup-to-disc ratio (hCDR) from segmentation masks.
        
        The hCDR is calculated using the maximum height method:
        - Find the column with maximum height for both OD and OC
        - Compute ratio of cup height to disc height
        
        Args:
            od: Optic disc segmentation (numpy array, H x W)
            oc: Optic cup segmentation (numpy array, H x W)
            
        Returns:
            float: Horizontal cup-to-disc ratio
        """
        # Compute the maximum height (vertical span) for cup and disc
        cup_diameter = self.compute_horizontal_diameter(oc)
        disc_diameter = self.compute_horizontal_diameter(od)
        
        # Return ratio with epsilon to prevent division by zero
        return cup_diameter / (disc_diameter + self.EPS)
    
    def compute_vCDR_error(self, pred_od, pred_oc, gt_od, gt_oc):
        """
        Compute vCDR prediction error, along with predicted vCDR and ground truth vCDR.
        
        Args:
            pred_od: Predicted optic disc segmentation
            pred_oc: Predicted optic cup segmentation
            gt_od: Ground truth optic disc segmentation
            gt_oc: Ground truth optic cup segmentation
            
        Returns:
            tuple: (vCDR_error, predicted_vCDR, ground_truth_vCDR)
        """
        pred_vCDR = self.vertical_cup_to_disc_ratio(pred_od, pred_oc)
        gt_vCDR = self.vertical_cup_to_disc_ratio(gt_od, gt_oc)
        vCDR_err = np.abs(gt_vCDR - pred_vCDR)
        return vCDR_err, pred_vCDR, gt_vCDR
    
    def compute_hCDR_error(self, pred_od, pred_oc, gt_od, gt_oc):
        """
        Compute hCDR prediction error, along with predicted hCDR and ground truth hCDR.
        
        Args:
            pred_od: Predicted optic disc segmentation
            pred_oc: Predicted optic cup segmentation
            gt_od: Ground truth optic disc segmentation
            gt_oc: Ground truth optic cup segmentation
            
        Returns:
            tuple: (hCDR_error, predicted_hCDR, ground_truth_hCDR)
        """
        pred_hCDR = self.horizontal_cup_to_disc_ratio(pred_od, pred_oc)
        gt_hCDR = self.horizontal_cup_to_disc_ratio(gt_od, gt_oc)
        hCDR_err = np.abs(gt_hCDR - pred_hCDR)
        return hCDR_err, pred_hCDR, gt_hCDR
    
    def classif_eval(self, classif_preds, classif_gts):
        """
        Compute AUC classification score.
        
        Args:
            classif_preds: Classification predictions
            classif_gts: Ground truth classification labels
            
        Returns:
            float: AUC score
        """
        auc = roc_auc_score(classif_gts, classif_preds)
        return auc
    
    def refine_seg(self, pred):
        """
        Only retain the biggest connected component of a segmentation map.
        
        Args:
            pred: Predicted segmentation tensor with shape (batch_size, height, width)
            
        Returns:
            torch.Tensor: Refined segmentation with only the largest connected component
                         for each sample in the batch
        """
        np_pred = pred.numpy()
            
        largest_ccs = []
        for i in range(np_pred.shape[0]):
            labeled, ncomponents = label(np_pred[i,:,:])
            bincounts = np.bincount(labeled.flat)[1:]
            if len(bincounts) == 0:
                largest_cc = labeled == 0
            else:
                largest_cc = labeled == np.argmax(bincounts)+1
            largest_cc = torch.tensor(largest_cc, dtype=torch.float32)
            largest_ccs.append(largest_cc)
        largest_ccs = torch.stack(largest_ccs)
        
        return largest_ccs


# Example usage:
# calculator = CalculationUNET(eps=1e-7)
# dice_score = calculator.compute_dice_coef(predictions, targets)
# auc_score = calculator.classif_eval(pred_labels, true_labels)
# refined_predictions = calculator.refine_seg(predictions)

# CDR calculations:
# vCDR = calculator.vertical_cup_to_disc_ratio(disc_seg, cup_seg)  # Uses max width method
# hCDR = calculator.horizontal_cup_to_disc_ratio(disc_seg, cup_seg)  # Uses max height method
# vCDR_error, pred_vCDR, gt_vCDR = calculator.compute_vCDR_error(pred_od, pred_oc, gt_od, gt_oc)
# hCDR_error, pred_hCDR, gt_hCDR = calculator.compute_hCDR_error(pred_od, pred_oc, gt_od, gt_oc)