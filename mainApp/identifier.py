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

# Assuming these are your existing imports
# from models import UNet, load_model
# from dataPreprocessor import preprocess_image
# from calculationTerms import CalculationUNET


class EnhancedGlaucomaEvaluator:
    """
    Enhanced evaluator that tracks misclassified samples in detail.
    """
    
    def __init__(self, model, device="cuda", vcdr_threshold=0.6, hcdr_threshold=0.6, output_size=(256, 256)):
        """
        Initialize the enhanced evaluator.
        """
        self.model = model
        self.device = device
        self.vcdr_threshold = vcdr_threshold
        self.hcdr_threshold = hcdr_threshold
        self.output_size = output_size
        # self.calculator = CalculationUNET()  # Uncomment when using actual calculator
        self.results = {}
        self.misclassified_samples = {}
        
    def analyze_misclassifications(self, results_df):
        """
        Analyze and categorize misclassified samples.
        
        Args:
            results_df: DataFrame with evaluation results
            
        Returns:
            dict: Categorized misclassified samples
        """
        # Extract image names from full paths
        results_df['image_name'] = results_df['img_path'].apply(lambda x: os.path.basename(x))
        
        # Categorize samples
        true_positives = results_df[(results_df['gt_glaucoma'] == 1) & (results_df['pred_glaucoma'] == 1)]
        true_negatives = results_df[(results_df['gt_glaucoma'] == 0) & (results_df['pred_glaucoma'] == 0)]
        false_positives = results_df[(results_df['gt_glaucoma'] == 0) & (results_df['pred_glaucoma'] == 1)]
        false_negatives = results_df[(results_df['gt_glaucoma'] == 1) & (results_df['pred_glaucoma'] == 0)]
        
        misclassification_analysis = {
            'false_positives': {
                'count': len(false_positives),
                'image_names': false_positives['image_name'].tolist(),
                'details': false_positives[['image_name', 'gt_vcdr', 'gt_hcdr', 'pred_vcdr', 'pred_hcdr', 
                                          'vcdr_error', 'hcdr_error', 'od_iou', 'oc_iou']].to_dict('records')
            },
            'false_negatives': {
                'count': len(false_negatives),
                'image_names': false_negatives['image_name'].tolist(),
                'details': false_negatives[['image_name', 'gt_vcdr', 'gt_hcdr', 'pred_vcdr', 'pred_hcdr', 
                                          'vcdr_error', 'hcdr_error', 'od_iou', 'oc_iou']].to_dict('records')
            },
            'true_positives': {
                'count': len(true_positives),
                'image_names': true_positives['image_name'].tolist(),
            },
            'true_negatives': {
                'count': len(true_negatives),
                'image_names': true_negatives['image_name'].tolist(),
            }
        }
        
        return misclassification_analysis
    
    def print_misclassification_report(self, misclassification_analysis):
        """
        Print detailed report of misclassified samples.
        """
        print("\n" + "="*80)
        print("DETAILED MISCLASSIFICATION ANALYSIS")
        print("="*80)
        
        # False Positives (Non-Glaucoma predicted as Glaucoma)
        fp_data = misclassification_analysis['false_positives']
        print(f"\n🔴 FALSE POSITIVES (Non-Glaucoma misclassified as Glaucoma): {fp_data['count']} cases")
        print("These are healthy eyes incorrectly classified as having glaucoma:")
        print("-" * 60)
        
        if fp_data['count'] > 0:
            for i, detail in enumerate(fp_data['details'][:10], 1):  # Show first 10
                print(f"{i:2d}. {detail['image_name']}")
                print(f"    GT CDR: vCDR={detail['gt_vcdr']:.3f}, hCDR={detail['gt_hcdr']:.3f}")
                print(f"    Pred CDR: vCDR={detail['pred_vcdr']:.3f}, hCDR={detail['pred_hcdr']:.3f}")
                print(f"    CDR Errors: vCDR={detail['vcdr_error']:.3f}, hCDR={detail['hcdr_error']:.3f}")
                print(f"    Segmentation: OD_IoU={detail['od_iou']:.3f}, OC_IoU={detail['oc_iou']:.3f}")
                print()
            
            if fp_data['count'] > 10:
                print(f"... and {fp_data['count'] - 10} more false positive cases.")
            
            print("All False Positive Image Names:")
            print(", ".join(fp_data['image_names']))
        
        # False Negatives (Glaucoma predicted as Non-Glaucoma)
        fn_data = misclassification_analysis['false_negatives']
        print(f"\n🟡 FALSE NEGATIVES (Glaucoma misclassified as Non-Glaucoma): {fn_data['count']} cases")
        print("These are glaucomatous eyes incorrectly classified as healthy:")
        print("-" * 60)
        
        if fn_data['count'] > 0:
            for i, detail in enumerate(fn_data['details'][:10], 1):  # Show first 10
                print(f"{i:2d}. {detail['image_name']}")
                print(f"    GT CDR: vCDR={detail['gt_vcdr']:.3f}, hCDR={detail['gt_hcdr']:.3f}")
                print(f"    Pred CDR: vCDR={detail['pred_vcdr']:.3f}, hCDR={detail['pred_hcdr']:.3f}")
                print(f"    CDR Errors: vCDR={detail['vcdr_error']:.3f}, hCDR={detail['hcdr_error']:.3f}")
                print(f"    Segmentation: OD_IoU={detail['od_iou']:.3f}, OC_IoU={detail['oc_iou']:.3f}")
                print()
            
            if fn_data['count'] > 10:
                print(f"... and {fn_data['count'] - 10} more false negative cases.")
            
            print("All False Negative Image Names:")
            print(", ".join(fn_data['image_names']))
        
        print(f"\n✅ CORRECT CLASSIFICATIONS:")
        print(f"True Positives (Glaucoma correctly identified): {misclassification_analysis['true_positives']['count']}")
        print(f"True Negatives (Non-Glaucoma correctly identified): {misclassification_analysis['true_negatives']['count']}")
        
        print("="*80)
    
    def save_misclassification_lists(self, misclassification_analysis, save_dir="misclassification_analysis"):
        """
        Save lists of misclassified images to separate files.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save False Positives
        fp_file = os.path.join(save_dir, "false_positives.txt")
        with open(fp_file, 'w') as f:
            f.write("FALSE POSITIVES (Non-Glaucoma predicted as Glaucoma)\n")
            f.write("="*60 + "\n")
            f.write(f"Total Count: {misclassification_analysis['false_positives']['count']}\n\n")
            f.write("Image Names:\n")
            for name in misclassification_analysis['false_positives']['image_names']:
                f.write(f"{name}\n")
        
        # Save False Negatives
        fn_file = os.path.join(save_dir, "false_negatives.txt")
        with open(fn_file, 'w') as f:
            f.write("FALSE NEGATIVES (Glaucoma predicted as Non-Glaucoma)\n")
            f.write("="*60 + "\n")
            f.write(f"Total Count: {misclassification_analysis['false_negatives']['count']}\n\n")
            f.write("Image Names:\n")
            for name in misclassification_analysis['false_negatives']['image_names']:
                f.write(f"{name}\n")
        
        # Save detailed analysis as CSV
        fp_df = pd.DataFrame(misclassification_analysis['false_positives']['details'])
        fn_df = pd.DataFrame(misclassification_analysis['false_negatives']['details'])
        
        if not fp_df.empty:
            fp_df.to_csv(os.path.join(save_dir, "false_positives_detailed.csv"), index=False)
        if not fn_df.empty:
            fn_df.to_csv(os.path.join(save_dir, "false_negatives_detailed.csv"), index=False)
        
        # Save summary report
        summary_file = os.path.join(save_dir, "misclassification_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("MISCLASSIFICATION SUMMARY\n")
            f.write("="*40 + "\n\n")
            f.write(f"False Positives: {misclassification_analysis['false_positives']['count']}\n")
            f.write(f"False Negatives: {misclassification_analysis['false_negatives']['count']}\n")
            f.write(f"True Positives: {misclassification_analysis['true_positives']['count']}\n")
            f.write(f"True Negatives: {misclassification_analysis['true_negatives']['count']}\n\n")
            
            total = sum([misclassification_analysis[key]['count'] for key in 
                        ['false_positives', 'false_negatives', 'true_positives', 'true_negatives']])
            f.write(f"Total Samples: {total}\n")
            f.write(f"Accuracy: {(misclassification_analysis['true_positives']['count'] + misclassification_analysis['true_negatives']['count']) / total:.4f}\n")
        
        print(f"\nMisclassification analysis saved to '{save_dir}' directory:")
        print(f"  - false_positives.txt: List of FP image names")
        print(f"  - false_negatives.txt: List of FN image names") 
        print(f"  - false_positives_detailed.csv: Detailed FP analysis")
        print(f"  - false_negatives_detailed.csv: Detailed FN analysis")
        print(f"  - misclassification_summary.txt: Overall summary")
    
    def plot_misclassification_analysis(self, misclassification_analysis, save_path=None):
        """
        Create visualizations for misclassification analysis.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: CDR Distribution by classification type
        fp_details = misclassification_analysis['false_positives']['details']
        fn_details = misclassification_analysis['false_negatives']['details']
        
        if fp_details and fn_details:
            fp_vcdr = [d['pred_vcdr'] for d in fp_details]
            fp_hcdr = [d['pred_hcdr'] for d in fp_details]
            fn_vcdr = [d['pred_vcdr'] for d in fn_details]
            fn_hcdr = [d['pred_hcdr'] for d in fn_details]
            
            axes[0, 0].scatter(fp_vcdr, fp_hcdr, color='red', alpha=0.7, label='False Positives', s=50)
            axes[0, 0].scatter(fn_vcdr, fn_hcdr, color='orange', alpha=0.7, label='False Negatives', s=50)
            axes[0, 0].axhline(y=self.hcdr_threshold, color='gray', linestyle='--', alpha=0.5)
            axes[0, 0].axvline(x=self.vcdr_threshold, color='gray', linestyle='--', alpha=0.5)
            axes[0, 0].set_xlabel('Predicted vCDR')
            axes[0, 0].set_ylabel('Predicted hCDR')
            axes[0, 0].set_title('Misclassified Samples in CDR Space')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: CDR Error Distribution
        if fp_details and fn_details:
            fp_vcdr_err = [d['vcdr_error'] for d in fp_details]
            fn_vcdr_err = [d['vcdr_error'] for d in fn_details]
            
            axes[0, 1].hist(fp_vcdr_err, bins=15, alpha=0.7, color='red', label='FP vCDR Error')
            axes[0, 1].hist(fn_vcdr_err, bins=15, alpha=0.7, color='orange', label='FN vCDR Error')
            axes[0, 1].set_xlabel('vCDR Error')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('CDR Error Distribution for Misclassified Samples')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Segmentation Quality for Misclassified Samples
        if fp_details and fn_details:
            fp_od_iou = [d['od_iou'] for d in fp_details]
            fp_oc_iou = [d['oc_iou'] for d in fp_details]
            fn_od_iou = [d['od_iou'] for d in fn_details]
            fn_oc_iou = [d['oc_iou'] for d in fn_details]
            
            axes[1, 0].scatter(fp_od_iou, fp_oc_iou, color='red', alpha=0.7, label='False Positives', s=50)
            axes[1, 0].scatter(fn_od_iou, fn_oc_iou, color='orange', alpha=0.7, label='False Negatives', s=50)
            axes[1, 0].set_xlabel('OD IoU')
            axes[1, 0].set_ylabel('OC IoU')
            axes[1, 0].set_title('Segmentation Quality for Misclassified Samples')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Classification Summary
        categories = ['True\nPositives', 'True\nNegatives', 'False\nPositives', 'False\nNegatives']
        counts = [
            misclassification_analysis['true_positives']['count'],
            misclassification_analysis['true_negatives']['count'],
            misclassification_analysis['false_positives']['count'],
            misclassification_analysis['false_negatives']['count']
        ]
        colors = ['green', 'lightgreen', 'red', 'orange']
        
        bars = axes[1, 1].bar(categories, counts, color=colors, alpha=0.7)
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Classification Results Summary')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                           str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def analyze_existing_results(results_csv_path):
    """
    Analyze misclassifications from existing evaluation results CSV.
    
    Args:
        results_csv_path: Path to the detailed_results.csv file from previous evaluation
    """
    # Load existing results
    try:
        results_df = pd.read_csv(results_csv_path)
        print(f"Loaded {len(results_df)} samples from {results_csv_path}")
    except Exception as e:
        print(f"Error loading results: {e}")
        return
    
    # Create a dummy evaluator just for analysis
    class DummyEvaluator:
        def __init__(self):
            self.vcdr_threshold = 0.6
            self.hcdr_threshold = 0.6
    
    evaluator = DummyEvaluator()
    enhanced_eval = EnhancedGlaucomaEvaluator(None, vcdr_threshold=0.6, hcdr_threshold=0.6)
    
    # Analyze misclassifications
    misclassification_analysis = enhanced_eval.analyze_misclassifications(results_df)
    
    # Print detailed report
    enhanced_eval.print_misclassification_report(misclassification_analysis)
    
    # Save analysis
    enhanced_eval.save_misclassification_lists(misclassification_analysis)
    
    # Create plots
    enhanced_eval.plot_misclassification_analysis(misclassification_analysis, 
                                                  save_path="misclassification_analysis.png")
    
    return misclassification_analysis


# Example usage
if __name__ == "__main__":
    # If you have existing results CSV file from your previous evaluation
    results_csv_path = "evaluation_results/detailed_results.csv"  # Update this path
    
    if os.path.exists(results_csv_path):
        print("Analyzing existing results...")
        analysis = analyze_existing_results(results_csv_path)
    else:
        print(f"Results file not found at {results_csv_path}")
        print("Please update the path or run the main evaluation first.")