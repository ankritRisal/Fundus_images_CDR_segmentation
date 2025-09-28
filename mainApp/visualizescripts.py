"""
Enhanced Glaucoma Analysis Dashboard with Contour Visualization
============================================================
A comprehensive Streamlit application for glaucoma risk analysis with AI-powered 
segmentation, contour visualization, and database management.
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sqlite3
import zipfile
from pathlib import Path
import tempfile
import shutil
import uuid
import csv
from typing import List, Dict, Any, Tuple, Optional

# Import your modules (assuming they're in the same directory)
from app import UnetOutput
from models import UNet, load_model
from calculationTerms import CalculationUNET

# =============================================================================
# CONFIGURATION
# =============================================================================

# Configure the page
st.set_page_config(
    page_title="Glaucoma Risk Analysis & Classification Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database paths
DOCTORS_DB = "doctors_database.db"
SINGLE_ANALYSIS_DB = "single_analysis_database.db"
BATCH_ANALYSIS_DB = "batch_analysis_database.db"

# =============================================================================
# CSS STYLING
# =============================================================================

def apply_custom_css():
    """Apply custom CSS with table styling and visual enhancements"""
    st.markdown("""
    <style>
        /* Main styling */
        .main-header {
            font-size: 2.5rem;
            color: #1f4e79;
            text-align: center;
            margin-bottom: 2rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        
        .doctor-portal {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        }
        
        .risk-high { 
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%); 
            color: white; padding: 1rem; border-radius: 15px; 
            text-align: center; font-weight: bold; 
        }
        .risk-moderate { 
            background: linear-gradient(135deg, #ffa726 0%, #fb8c00 100%); 
            color: white; padding: 1rem; border-radius: 15px; 
            text-align: center; font-weight: bold; 
        }
        .risk-low { 
            background: linear-gradient(135deg, #66bb6a 0%, #43a047 100%); 
            color: white; padding: 1rem; border-radius: 15px; 
            text-align: center; font-weight: bold; 
        }
        
        .classification-positive { 
            background: #ffcdd2; color: #d32f2f; padding: 0.5rem; 
            border-radius: 8px; font-weight: bold; text-align: center; 
        }
        .classification-negative { 
            background: #c8e6c9; color: #388e3c; padding: 0.5rem; 
            border-radius: 8px; font-weight: bold; text-align: center; 
        }
        
        /* Table styling */
        div[data-testid="stDataFrame"] {
            background-color: white !important;
        }
        
        div[data-testid="stDataFrame"] table {
            background-color: white !important;
            color: #000000 !important;
            border-collapse: collapse !important;
            width: 100% !important;
        }
        
        div[data-testid="stDataFrame"] table th {
            background-color: #f8f9fa !important;
            color: #000000 !important;
            font-weight: bold !important;
            padding: 12px 8px !important;
            border: 1px solid #dee2e6 !important;
            text-align: left !important;
        }
        
        div[data-testid="stDataFrame"] table td {
            background-color: white !important;
            color: #000000 !important;
            padding: 8px !important;
            border: 1px solid #dee2e6 !important;
            text-align: left !important;
        }
        
        /* Analysis visualization styling */
        .analysis-container {
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            background-color: #fafafa;
        }
        
        .glaucoma-row {
            background-color: #ffebee !important;
        }
        
        .glaucoma-row td {
            background-color: #ffebee !important;
        }
    </style>
    """, unsafe_allow_html=True)

# Apply CSS immediately
apply_custom_css()

# =============================================================================
# DATABASE MANAGEMENT
# =============================================================================

class DatabaseManager:
    """Centralized database management without image storage"""
    
    @staticmethod
    def init_all_databases():
        """Initialize all databases at once"""
        DatabaseManager.init_doctors_database()
        DatabaseManager.init_single_analysis_database() 
        DatabaseManager.init_batch_analysis_database()
    
    @staticmethod
    def init_doctors_database():
        """Initialize doctors database"""
        conn = sqlite3.connect(DOCTORS_DB)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS doctors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doctor_name TEXT NOT NULL,
                doctor_id TEXT UNIQUE NOT NULL,
                threshold_value REAL DEFAULT 0.6,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    
    @staticmethod
    def init_single_analysis_database():
        """Initialize single analysis results database (table format)"""
        conn = sqlite3.connect(SINGLE_ANALYSIS_DB)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS single_analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doctor_id TEXT NOT NULL,
                doctor_name TEXT NOT NULL,
                threshold_value REAL NOT NULL,
                patient_id TEXT NOT NULL,
                patient_name TEXT,
                image_name TEXT NOT NULL,
                ai_vertical_cdr REAL,
                ai_horizontal_cdr REAL,
                classification TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                doctor_notes TEXT,
                analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doctor_id) REFERENCES doctors (doctor_id)
            )
        ''')
        conn.commit()
        conn.close()
    
    @staticmethod
    def init_batch_analysis_database():
        """Initialize batch analysis results database"""
        conn = sqlite3.connect(BATCH_ANALYSIS_DB)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS batch_analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id TEXT NOT NULL,
                doctor_id TEXT NOT NULL,
                doctor_name TEXT NOT NULL,
                threshold_value REAL NOT NULL,
                patient_id TEXT NOT NULL,
                patient_name TEXT,
                image_name TEXT NOT NULL,
                ai_vertical_cdr REAL,
                ai_horizontal_cdr REAL,
                classification TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doctor_id) REFERENCES doctors (doctor_id)
            )
        ''')
        conn.commit()
        conn.close()
    
    @staticmethod
    def execute_query(db_path: str, query: str, params: tuple = None) -> List[Dict]:
        """Generic database query executor"""
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
            
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results
    
    @staticmethod
    def insert_record(db_path: str, table: str, data: Dict) -> bool:
        """Generic record insertion"""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            columns = list(data.keys())
            placeholders = ['?' for _ in columns]
            values = list(data.values())
            
            query = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
            cursor.execute(query, values)
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            st.error(f"Database error: {str(e)}")
            return False
    
    @staticmethod
    def get_all_doctors() -> List[Dict]:
        """Get all doctors"""
        return DatabaseManager.execute_query(DOCTORS_DB, 'SELECT * FROM doctors ORDER BY doctor_name')
    
    @staticmethod
    def get_doctor(doctor_id: str) -> Optional[Dict]:
        """Get specific doctor"""
        results = DatabaseManager.execute_query(
            DOCTORS_DB, 'SELECT * FROM doctors WHERE doctor_id = ?', (doctor_id,)
        )
        return results[0] if results else None

# =============================================================================
# CONTOUR VISUALIZATION MANAGER
# =============================================================================

class ContourVisualizationManager:
    """Manages contour visualization using the exact app.py method"""
    
    @staticmethod
    def generate_contour_visualization(analysis_results: Dict, original_image: Image.Image, 
                                     threshold: float, show_plot: bool = False) -> plt.Figure:
        """Generate 3-panel contour visualization using app.py's exact method"""
        try:
            # Use the results directly from app.py predict_segmentation
            results = analysis_results['ai_results']
            
            # Create figure with 3 panels (remove the prediction masks panel)
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Prepare image for display (same as app.py)
            if results["input_image"].shape[0] == 3:  # If channels first
                display_image = results["input_image"].transpose(1, 2, 0)
            else:
                display_image = results["input_image"]
            
            # Plot 1: Original Fundus Image
            axes[0].imshow(display_image)
            axes[0].set_title('Original Fundus Image', fontsize=14, fontweight='bold', pad=10)
            axes[0].axis('off')
            
            # Get prediction masks (same as app.py)
            pred_od_np = results["pred_od"]
            pred_oc_np = results["pred_oc"]
            
            # Get diameters (same as app.py)
            od_vertical_diameter = results["od_vertical_diameter"]
            oc_vertical_diameter = results["oc_vertical_diameter"] 
            od_horizontal_diameter = results["od_horizontal_diameter"]
            oc_horizontal_diameter = results["oc_horizontal_diameter"]
            
            # Plot 2: Vertical CDR Analysis (copied exactly from app.py)
            axes[1].imshow(display_image)
            
            # Draw prediction contours (exact copy from app.py)
            if results["od_contours"]:
                for contour in results["od_contours"]:
                    axes[1].plot(contour[:, 1], contour[:, 0], 'red', linewidth=3, alpha=0.9)
            
            if results["oc_contours"]:
                for contour in results["oc_contours"]:
                    axes[1].plot(contour[:, 1], contour[:, 0], 'blue', linewidth=3, alpha=0.9)
            
            # Find the column with maximum height for vertical measurement (exact copy)
            if np.any(pred_od_np > 0):
                od_col_heights = np.sum(pred_od_np > 0, axis=0)
                max_od_col = np.argmax(od_col_heights)
                od_col_indices = np.where(pred_od_np[:, max_od_col] > 0)[0]
                
                if len(od_col_indices) > 0:
                    od_top = od_col_indices[0]
                    od_bottom = od_col_indices[-1]
                    
                    # Draw VERTICAL line for vertical diameter (height) - exact copy
                    axes[1].plot([max_od_col, max_od_col], [od_top, od_bottom], 
                                'black', linewidth=4, alpha=0.8)
                    
                    # Add arrows at the ends - exact copy
                    axes[1].annotate('', xy=(max_od_col, od_top), 
                                    xytext=(max_od_col, od_top-10),
                                    arrowprops=dict(arrowstyle='->', color='black', lw=2))
                    axes[1].annotate('', xy=(max_od_col, od_bottom), 
                                    xytext=(max_od_col, od_bottom+10),
                                    arrowprops=dict(arrowstyle='->', color='black', lw=2))
                    
                    axes[1].text(0.95, 0.95, f'OD: {od_vertical_diameter}px', 
                                transform=axes[1].transAxes,
                                fontsize=11, fontweight='bold', ha='right', va='top',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
            
            # Similar for OC vertical measurement - exact copy
            if np.any(pred_oc_np > 0):
                oc_col_heights = np.sum(pred_oc_np > 0, axis=0)
                max_oc_col = np.argmax(oc_col_heights)
                oc_col_indices = np.where(pred_oc_np[:, max_oc_col] > 0)[0]
                
                if len(oc_col_indices) > 0:
                    oc_top = oc_col_indices[0]
                    oc_bottom = oc_col_indices[-1]
                    
                    # Draw VERTICAL line for cup (offset to avoid overlap) - exact copy
                    axes[1].plot([max_oc_col+5, max_oc_col+5], [oc_top, oc_bottom], 
                                'navy', linewidth=4, alpha=0.8)
                    
                    axes[1].text(0.95, 0.85, f'OC: {oc_vertical_diameter}px', 
                                transform=axes[1].transAxes,
                                fontsize=11, fontweight='bold', ha='right', va='top',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.9))
            
            # Verify CDR calculation - exact copy
            calculated_vCDR = oc_vertical_diameter / (od_vertical_diameter + 1e-8)
            axes[1].text(0.5, -0.12, f'Predicted vCDR = {calculated_vCDR:.3f}', 
                        transform=axes[1].transAxes, fontsize=12, fontweight='bold',
                        ha='center', va='top',
                        bbox=dict(boxstyle="round,pad=0.4", facecolor='lightgreen', alpha=0.8))
            
            axes[1].set_title('Vertical CDR Analysis\n(Max Height Method)', fontsize=14, fontweight='bold', pad=10)
            axes[1].axis('off')
            
            # Plot 3: Horizontal CDR Analysis (copied exactly from app.py)
            axes[2].imshow(display_image)
            
            # Draw prediction contours - exact copy
            if results["od_contours"]:
                for contour in results["od_contours"]:
                    axes[2].plot(contour[:, 1], contour[:, 0], 'red', linewidth=3, alpha=0.9)
            
            if results["oc_contours"]:
                for contour in results["oc_contours"]:
                    axes[2].plot(contour[:, 1], contour[:, 0], 'blue', linewidth=3, alpha=0.9)
            
            # Find the row with maximum width for horizontal measurement - exact copy
            if np.any(pred_od_np > 0):
                od_row_widths = np.sum(pred_od_np > 0, axis=1)
                max_od_row = np.argmax(od_row_widths)
                od_row_indices = np.where(pred_od_np[max_od_row, :] > 0)[0]
                
                if len(od_row_indices) > 0:
                    od_left = od_row_indices[0]
                    od_right = od_row_indices[-1]
                    
                    # Draw HORIZONTAL line for horizontal diameter (width) - exact copy
                    axes[2].plot([od_left, od_right], [max_od_row, max_od_row], 
                                'black', linewidth=4, alpha=0.8)
                    
                    # Add arrows - exact copy
                    axes[2].annotate('', xy=(od_left, max_od_row), 
                                    xytext=(od_left-10, max_od_row),
                                    arrowprops=dict(arrowstyle='->', color='black', lw=2))
                    axes[2].annotate('', xy=(od_right, max_od_row), 
                                    xytext=(od_right+10, max_od_row),
                                    arrowprops=dict(arrowstyle='->', color='black', lw=2))
                    
                    axes[2].text(0.95, 0.95, f'OD: {od_horizontal_diameter}px', 
                                transform=axes[2].transAxes,
                                fontsize=11, fontweight='bold', ha='right', va='top',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
            
            # Similar for OC horizontal - exact copy
            if np.any(pred_oc_np > 0):
                oc_row_widths = np.sum(pred_oc_np > 0, axis=1)
                max_oc_row = np.argmax(oc_row_widths)
                oc_row_indices = np.where(pred_oc_np[max_oc_row, :] > 0)[0]
                
                if len(oc_row_indices) > 0:
                    oc_left = oc_row_indices[0]
                    oc_right = oc_row_indices[-1]
                    
                    # Draw HORIZONTAL line for cup (offset to avoid overlap) - exact copy
                    axes[2].plot([oc_left, oc_right], [max_oc_row-5, max_oc_row-5], 
                                'navy', linewidth=4, alpha=0.8)
                    
                    axes[2].text(0.95, 0.85, f'OC: {oc_horizontal_diameter}px', 
                                transform=axes[2].transAxes,
                                fontsize=11, fontweight='bold', ha='right', va='top',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.9))
            
            # Verify hCDR calculation - exact copy
            calculated_hCDR = oc_horizontal_diameter / (od_horizontal_diameter + 1e-8)
            axes[2].text(0.5, -0.12, f'Predicted hCDR = {calculated_hCDR:.3f}', 
                        transform=axes[2].transAxes, fontsize=12, fontweight='bold',
                        ha='center', va='top',
                        bbox=dict(boxstyle="round,pad=0.4", facecolor='lightyellow', alpha=0.8))
            
            axes[2].set_title('Horizontal CDR Analysis\n(Max Width Method)', fontsize=14, fontweight='bold', pad=10)
            axes[2].axis('off')
            
            # Overall assessment - adapted from app.py
            vCDR = results['pred_vCDR']
            hCDR = results['pred_hCDR']
            max_cdr = max(vCDR, hCDR)
            
            if max_cdr > threshold:
                risk_level = "HIGH RISK"
                cdr_bg = "mistyrose"
            elif max_cdr > threshold * 0.8:
                risk_level = "MODERATE RISK"
                cdr_bg = "peachpuff"
            else:
                risk_level = "LOW RISK"
                cdr_bg = "lightgreen"
            
            # Assessment text
            assessment_text = f'PREDICTED: vCDR = {vCDR:.3f} | hCDR = {hCDR:.3f} | Max CDR = {max_cdr:.3f} | Risk Level: {risk_level}'
            
            fig.text(0.5, 0.02, assessment_text, 
                    fontsize=12, fontweight='bold', ha='center', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor=cdr_bg, alpha=0.9))
            
            # Adjust layout - same as app.py
            plt.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.12, wspace=0.08)
            
            if show_plot:
                plt.show()
            
            return fig
            
        except Exception as e:
            st.error(f"Error generating contour visualization: {str(e)}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    @staticmethod
    def display_contour_visualization_in_streamlit(fig):
        """Display the matplotlib figure in Streamlit"""
        if fig is not None:
            st.pyplot(fig)
            plt.close(fig)  # Clean up memory


class AnalysisEngine:
    """Analysis processing with contour visualization support"""
    
    @staticmethod
    @st.cache_resource
    def load_glaucoma_model():
        """Load the trained U-Net model (cached for efficiency)"""
        try:
            device = torch.device("cpu")
            model = UNet(n_channels=3, n_classes=2)
            model_path = "/home/ankritrisal/Documents/project glaucoma /mainApp/binModel/best_seg.pth"
            
            if not os.path.exists(model_path):
                st.error(f"Model file '{model_path}' not found.")
                return None, None
                
            model = load_model(model, model_path, device)
            unet_op = UnetOutput(device)
            return model, unet_op
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None, None
    
    @staticmethod
    def process_single_image_with_contour_viz(image_path: str, original_image: Image.Image, 
                                            model, unet_op, doctor_info: Dict) -> Dict:
        """Process single image and generate contour visualization using exact app.py method"""
        # Use the app.py predict_segmentation method directly
        results = unet_op.predict_segmentation(model, image_path)
        
        classification = RiskAnalyzer.classify_glaucoma(
            results['pred_vCDR'], 
            results['pred_hCDR'], 
            doctor_info['threshold_value']
        )
        
        risk_level = RiskAnalyzer.assess_risk_level(
            results['pred_vCDR'], 
            results['pred_hCDR'], 
            doctor_info['threshold_value']
        )
        
        # Generate contour visualization using app.py's exact method
        contour_fig = ContourVisualizationManager.generate_contour_visualization(
            {'ai_results': results},
            original_image,
            doctor_info['threshold_value']
        )
        
        return {
            'ai_results': results,
            'classification': classification,
            'risk_level': risk_level,
            'contour_visualization': contour_fig
        }


# Keep the rest of the classes unchanged
class RiskAnalyzer:
    """Risk analysis utilities"""
    
    @staticmethod
    def classify_glaucoma(vcdr: float, hcdr: float, threshold: float) -> str:
        """Classify glaucoma based on doctor's threshold"""
        return "glaucoma" if max(vcdr, hcdr) > threshold else "non-glaucoma"
    
    @staticmethod
    def assess_risk_level(vcdr: float, hcdr: float, threshold: float) -> str:
        """Assess risk level based on CDR values and threshold"""
        max_cdr = max(vcdr, hcdr)
        if max_cdr > threshold:
            return "High"
        elif max_cdr > threshold * 0.8:
            return "Moderate"
        else:
            return "Low"

# =============================================================================
# UI COMPONENTS
# =============================================================================

class UIComponents:
    """Reusable UI components"""
    
    @staticmethod
    def display_metrics_row(metrics: Dict[str, Any]):
        """Display metrics in columns"""
        cols = st.columns(len(metrics))
        for i, (label, value) in enumerate(metrics.items()):
            with cols[i]:
                if isinstance(value, tuple):
                    st.metric(label, value[0], value[1])
                else:
                    st.metric(label, value)
    
    @staticmethod
    def create_styled_dataframe(df: pd.DataFrame, highlight_glaucoma: bool = True) -> None:
        """Create styled dataframe"""
        if df.empty:
            st.info("No data available")
            return
        
        # Format numerical columns
        for col in ['ai_vertical_cdr', 'ai_horizontal_cdr']:
            if col in df.columns:
                df[col] = df[col].round(3)
        
        st.dataframe(df, use_container_width=True, height=min(400, len(df) * 35 + 38))
    
    @staticmethod
    def display_filters(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
        """Unified filtering component"""
        st.subheader("🔍 Filters")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            risk_levels = st.multiselect(
                "Risk Level",
                options=['Low', 'Moderate', 'High'],
                default=['Low', 'Moderate', 'High'],
                key=f"risk_{key_prefix}"
            )
        
        with col2:
            classifications = st.multiselect(
                "Classification",
                options=['glaucoma', 'non-glaucoma'],
                default=['glaucoma', 'non-glaucoma'],
                key=f"class_{key_prefix}"
            )
        
        with col3:
            if 'doctor_name' in df.columns:
                doctors = df['doctor_name'].unique()
                selected_doctors = st.multiselect(
                    "Doctor",
                    options=doctors,
                    default=[],
                    key=f"doctor_{key_prefix}"
                )
            else:
                selected_doctors = []
        
        with col4:
            date_filter = st.selectbox(
                "Date Filter",
                options=["All Time", "Last Day", "Last Week", "Last Month"],
                key=f"date_{key_prefix}"
            )
        
        # Apply filters
        filtered_df = df.copy()
        
        if risk_levels:
            filtered_df = filtered_df[filtered_df['risk_level'].isin(risk_levels)]
        if classifications:
            filtered_df = filtered_df[filtered_df['classification'].isin(classifications)]
        if selected_doctors:
            filtered_df = filtered_df[filtered_df['doctor_name'].isin(selected_doctors)]
        
        # Date filtering
        if date_filter != "All Time":
            current_date = datetime.now()
            if date_filter == "Last Day":
                cutoff_date = current_date - timedelta(days=1)
            elif date_filter == "Last Week":
                cutoff_date = current_date - timedelta(weeks=1)
            else:  # Last Month
                cutoff_date = current_date - timedelta(days=30)
            
            filtered_df['analysis_date'] = pd.to_datetime(filtered_df['analysis_date'])
            filtered_df = filtered_df[filtered_df['analysis_date'] >= cutoff_date]
        
        st.info(f"Showing {len(filtered_df)} of {len(df)} records")
        return filtered_df

# =============================================================================
# PAGE COMPONENTS
# =============================================================================

def display_analysis_results(analysis_results, threshold):
    """Display analysis results with risk assessment"""
    st.success("Analysis completed!")
    
    # Classification
    classification = analysis_results['classification']
    if classification == "glaucoma":
        st.markdown('<div class="classification-positive">GLAUCOMA DETECTED</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="classification-negative">NON-GLAUCOMA</div>', unsafe_allow_html=True)
    
    # Risk assessment
    risk_level = analysis_results['risk_level']
    risk_class = f"risk-{risk_level.lower()}"
    risk_messages = {
        'High': f"HIGH RISK (CDR > {threshold})",
        'Moderate': f"MODERATE RISK (CDR close to {threshold})", 
        'Low': f"LOW RISK (CDR < {threshold * 0.8})"
    }
    st.markdown(f'<div class="{risk_class}">{risk_messages[risk_level]}</div>', unsafe_allow_html=True)
    
    # CDR metrics
    results = analysis_results['ai_results']
    metrics = {
        "Vertical CDR": f"{results['pred_vCDR']:.3f}",
        "Horizontal CDR": f"{results['pred_hCDR']:.3f}",
        "Max CDR": f"{max(results['pred_vCDR'], results['pred_hCDR']):.3f}",
        "Threshold": f"{threshold:.3f}"
    }
    UIComponents.display_metrics_row(metrics)

def display_clinical_dashboard():
    """Clinical dashboard with combined analytics"""
    st.subheader("Clinical Dashboard")
    
    try:
        single_results = pd.DataFrame(DatabaseManager.execute_query(
            SINGLE_ANALYSIS_DB, 'SELECT * FROM single_analysis_results ORDER BY analysis_date DESC'
        ))
        batch_results = pd.DataFrame(DatabaseManager.execute_query(
            BATCH_ANALYSIS_DB, 'SELECT * FROM batch_analysis_results ORDER BY analysis_date DESC'
        ))
        
        if single_results.empty and batch_results.empty:
            st.info("No analysis data available yet.")
            return
        
        # Combine results
        all_results = []
        if not single_results.empty:
            single_results['analysis_type'] = 'Single'
            all_results.append(single_results)
        if not batch_results.empty:
            batch_results['analysis_type'] = 'Batch'
            all_results.append(batch_results)
        
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # Summary metrics
        metrics = {
            "Total Analyses": len(combined_results),
            "Glaucoma Cases": len(combined_results[combined_results['classification'] == 'glaucoma']),
            "High Risk": len(combined_results[combined_results['risk_level'] == 'High']),
            "Unique Patients": combined_results['patient_id'].nunique(),
            "Active Doctors": combined_results['doctor_id'].nunique()
        }
        
        UIComponents.display_metrics_row(metrics)
        
        # Recent glaucoma cases
        st.subheader("Recent Glaucoma Cases")
        recent_glaucoma = combined_results[
            combined_results['classification'] == 'glaucoma'
        ].head(10)
        
        if not recent_glaucoma.empty:
            display_cols = ['doctor_name', 'patient_id', 'patient_name', 'ai_vertical_cdr', 
                          'ai_horizontal_cdr', 'risk_level', 'analysis_date']
            UIComponents.create_styled_dataframe(recent_glaucoma[display_cols], True)
        else:
            st.info("No recent glaucoma cases found.")
            
    except Exception as e:
        st.error(f"Error loading dashboard data: {str(e)}")

# =============================================================================
# MAIN APPLICATION PAGES
# =============================================================================

def doctor_portal_page():
    """Doctor portal with registration and dashboard"""
    st.markdown('<h1 class="main-header">Doctor Portal</h1>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Doctor Registration", "Dashboard"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="doctor-portal">', unsafe_allow_html=True)
            st.header("Register New Doctor")
            
            doctor_name = st.text_input("Doctor Name*")
            doctor_id = st.text_input("Doctor ID*")
            threshold_value = st.number_input("Threshold", min_value=0.1, max_value=1.0, value=0.6, step=0.01)
            
            if st.button("Register Doctor", type="primary"):
                if doctor_name and doctor_id:
                    data = {
                        'doctor_name': doctor_name,
                        'doctor_id': doctor_id,
                        'threshold_value': threshold_value
                    }
                    if DatabaseManager.insert_record(DOCTORS_DB, 'doctors', data):
                        st.success("Doctor registered successfully!")
                        st.balloons()
                    else:
                        st.error("Registration failed. ID may already exist.")
                else:
                    st.warning("Please fill in all required fields.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.subheader("Registered Doctors")
            doctors = DatabaseManager.get_all_doctors()
            if doctors:
                df = pd.DataFrame(doctors)[['doctor_name', 'doctor_id', 'threshold_value', 'created_at']]
                UIComponents.create_styled_dataframe(df, False)
            else:
                st.info("No doctors registered yet.")
    
    with tab2:
        display_clinical_dashboard()

def single_analysis_page():
    """Enhanced single analysis with contour visualization"""
    st.header("Single Image Analysis")
    
    # Doctor selection
    doctors = DatabaseManager.get_all_doctors()
    if not doctors:
        st.error("No doctors registered. Please register a doctor first.")
        return
    
    doctor_options = {f"{doc['doctor_name']} ({doc['doctor_id']})": doc for doc in doctors}
    selected_doctor_display = st.selectbox("Select Doctor*", options=list(doctor_options.keys()))
    selected_doctor = doctor_options[selected_doctor_display]
    
    st.info(f"Current Threshold: **{selected_doctor['threshold_value']}** CDR")
    
    # Load model
    with st.spinner("Loading AI model..."):
        model, unet_op = AnalysisEngine.load_glaucoma_model()
    
    if not model:
        return
    
    st.success("Model loaded successfully!")
    
    # Patient information
    col1, col2 = st.columns(2)
    with col1:
        patient_id = st.text_input("Patient ID*")
        patient_name = st.text_input("Patient Name")
    with col2:
        doctor_notes = st.text_area("Doctor's Notes")
    
    if not patient_id:
        st.warning("Please enter a Patient ID.")
        return
    
    # Image upload
    uploaded_file = st.file_uploader(
        "Choose a fundus image...",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif']
    )
    
    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(image, caption="Uploaded Image", width=400)
        except Exception as e:
            st.error(f"Invalid image: {str(e)}")
            return
        
        if st.button("Analyze Image", type="primary"):
            with st.spinner("Analyzing..."):
                try:
                    # Save temp file
                    temp_path = f"temp_single_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    image.save(temp_path)
                    
                    # Process with contour visualization
                    analysis_results = AnalysisEngine.process_single_image_with_contour_viz(
                        temp_path, image, model, unet_op, selected_doctor
                    )
                    
                    # Clean up temp file
                    os.remove(temp_path)
                    
                    # Save to database (no image path needed)
                    data = {
                        'doctor_id': selected_doctor['doctor_id'],
                        'doctor_name': selected_doctor['doctor_name'],
                        'threshold_value': selected_doctor['threshold_value'],
                        'patient_id': patient_id,
                        'patient_name': patient_name,
                        'image_name': uploaded_file.name,
                        'ai_vertical_cdr': analysis_results['ai_results']['pred_vCDR'],
                        'ai_horizontal_cdr': analysis_results['ai_results']['pred_hCDR'],
                        'classification': analysis_results['classification'],
                        'risk_level': analysis_results['risk_level'],
                        'doctor_notes': doctor_notes
                    }
                    
                    DatabaseManager.insert_record(SINGLE_ANALYSIS_DB, 'single_analysis_results', data)
                    
                    # Display results
                    display_analysis_results(analysis_results, selected_doctor['threshold_value'])
                    
                    # Display the contour visualization
                    if analysis_results.get('contour_visualization'):
                        st.subheader("Detailed Contour Analysis Visualization")
                        st.markdown('<div class="analysis-container">', unsafe_allow_html=True)
                        ContourVisualizationManager.display_contour_visualization_in_streamlit(
                            analysis_results['contour_visualization']
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Analysis error: {str(e)}")

def batch_analysis_page():
    """Batch analysis page"""
    st.header("Batch Image Analysis")
    
    # Doctor selection
    doctors = DatabaseManager.get_all_doctors()
    if not doctors:
        st.error("No doctors registered. Please register a doctor first.")
        return
    
    doctor_options = {f"{doc['doctor_name']} ({doc['doctor_id']})": doc for doc in doctors}
    selected_doctor_display = st.selectbox("Select Doctor*", options=list(doctor_options.keys()))
    selected_doctor = doctor_options[selected_doctor_display]
    
    st.info(f"Current Threshold: **{selected_doctor['threshold_value']}** CDR")
    
    # Load model
    with st.spinner("Loading AI model..."):
        model, unet_op = AnalysisEngine.load_glaucoma_model()
    
    if not model:
        return
    
    st.success("Model loaded successfully!")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Select multiple images:",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.info(f"Selected {len(uploaded_files)} files")
        
        # Patient ID options
        auto_generate = st.radio(
            "Patient ID Method:",
            ["Use filenames", "Auto-generate IDs"]
        )
        
        if auto_generate == "Auto-generate IDs":
            id_prefix = st.text_input("ID Prefix:", value="BATCH")
        
        if st.button("Process Batch", type="primary"):
            process_batch_analysis(uploaded_files, model, unet_op, selected_doctor, 
                                 auto_generate, id_prefix if auto_generate == "Auto-generate IDs" else None)

def process_batch_analysis(uploaded_files, model, unet_op, selected_doctor, auto_generate, id_prefix):
    """Process batch analysis"""
    batch_id = f"BATCH_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    successful_analyses = []
    failed_analyses = []
    
    for idx, uploaded_file in enumerate(uploaded_files):
        try:
            status_text.text(f"Processing {uploaded_file.name} ({idx+1}/{len(uploaded_files)})")
            
            # Generate patient ID
            if auto_generate == "Auto-generate IDs":
                patient_id = f"{id_prefix}_{idx+1:03d}"
                patient_name = os.path.splitext(uploaded_file.name)[0]
            else:
                patient_id = os.path.splitext(uploaded_file.name)[0]
                patient_name = ""
            
            # Process image
            image = Image.open(uploaded_file)
            temp_path = f"temp_batch_{idx}.jpg"
            image.save(temp_path)
            
            # Simple processing without visualization for batch
            results = unet_op.predict_segmentation(model, temp_path)
            classification = RiskAnalyzer.classify_glaucoma(
                results['pred_vCDR'], results['pred_hCDR'], selected_doctor['threshold_value']
            )
            risk_level = RiskAnalyzer.assess_risk_level(
                results['pred_vCDR'], results['pred_hCDR'], selected_doctor['threshold_value']
            )
            
            os.remove(temp_path)
            
            # Save to database
            data = {
                'batch_id': batch_id,
                'doctor_id': selected_doctor['doctor_id'],
                'doctor_name': selected_doctor['doctor_name'],
                'threshold_value': selected_doctor['threshold_value'],
                'patient_id': patient_id,
                'patient_name': patient_name,
                'image_name': uploaded_file.name,
                'ai_vertical_cdr': results['pred_vCDR'],
                'ai_horizontal_cdr': results['pred_hCDR'],
                'classification': classification,
                'risk_level': risk_level
            }
            
            DatabaseManager.insert_record(BATCH_ANALYSIS_DB, 'batch_analysis_results', data)
            
            successful_analyses.append({
                'patient_id': patient_id,
                'classification': classification,
                'risk_level': risk_level,
                'vcdr': results['pred_vCDR'],
                'hcdr': results['pred_hCDR']
            })
            
        except Exception as e:
            failed_analyses.append({
                'patient_id': patient_id if 'patient_id' in locals() else uploaded_file.name,
                'error': str(e)
            })
        
        progress_bar.progress((idx + 1) / len(uploaded_files))
    
    status_text.success(f"Batch complete! {len(successful_analyses)}/{len(uploaded_files)} processed successfully.")
    
    # Display results
    if successful_analyses:
        results_df = pd.DataFrame(successful_analyses)
        
        # Summary metrics
        glaucoma_cases = len(results_df[results_df['classification'] == 'glaucoma'])
        high_risk_cases = len(results_df[results_df['risk_level'] == 'High'])
        
        metrics = {
            "Successfully Processed": len(successful_analyses),
            "Glaucoma Detected": glaucoma_cases,
            "High Risk Cases": high_risk_cases,
            "Failed": len(failed_analyses)
        }
        
        UIComponents.display_metrics_row(metrics)
        
        # Results table
        display_columns = ['patient_id', 'classification', 'risk_level', 'vcdr', 'hcdr']
        UIComponents.create_styled_dataframe(results_df[display_columns])
        
        # Download option
        csv_data = results_df.to_csv(index=False)
        st.download_button(
            label="Download Results (CSV)",
            data=csv_data,
            file_name=f"batch_results_{batch_id}.csv",
            mime="text/csv"
        )
    
    # Show failures if any
    if failed_analyses:
        st.subheader("Failed Analyses")
        failed_df = pd.DataFrame(failed_analyses)
        UIComponents.create_styled_dataframe(failed_df, False)

def results_database_page():
    """Enhanced results database with table format"""
    st.header("Results Database")
    
    tab1, tab2, tab3 = st.tabs(["Single Results", "Batch Results", "Analytics"])
    
    with tab1:
        st.subheader("Single Analysis Results")
        try:
            results_df = pd.DataFrame(DatabaseManager.execute_query(
                SINGLE_ANALYSIS_DB, 'SELECT * FROM single_analysis_results ORDER BY analysis_date DESC'
            ))
            
            if not results_df.empty:
                # Summary metrics
                metrics = {
                    "Total Records": len(results_df),
                    "Glaucoma Cases": len(results_df[results_df['classification'] == 'glaucoma']),
                    "High Risk": len(results_df[results_df['risk_level'] == 'High']),
                    "Unique Patients": results_df['patient_id'].nunique()
                }
                UIComponents.display_metrics_row(metrics)
                
                # Apply filters
                filtered_df = UIComponents.display_filters(results_df, "single")
                
                # Display table
                display_cols = ['doctor_name', 'patient_id', 'patient_name', 'image_name',
                              'ai_vertical_cdr', 'ai_horizontal_cdr', 'classification', 
                              'risk_level', 'doctor_notes', 'analysis_date']
                UIComponents.create_styled_dataframe(filtered_df[display_cols])
                
            else:
                st.info("No single analysis results found.")
                
        except Exception as e:
            st.error(f"Error loading single analysis data: {str(e)}")
    
    with tab2:
        st.subheader("Batch Analysis Results")
        try:
            results_df = pd.DataFrame(DatabaseManager.execute_query(
                BATCH_ANALYSIS_DB, 'SELECT * FROM batch_analysis_results ORDER BY analysis_date DESC'
            ))
            
            if not results_df.empty:
                # Summary metrics
                metrics = {
                    "Total Records": len(results_df),
                    "Glaucoma Cases": len(results_df[results_df['classification'] == 'glaucoma']),
                    "High Risk": len(results_df[results_df['risk_level'] == 'High']),
                    "Unique Batches": results_df['batch_id'].nunique()
                }
                UIComponents.display_metrics_row(metrics)
                
                # Apply filters
                filtered_df = UIComponents.display_filters(results_df, "batch")
                
                # Display table
                display_cols = ['batch_id', 'doctor_name', 'patient_id', 'patient_name', 
                              'ai_vertical_cdr', 'ai_horizontal_cdr', 'classification', 
                              'risk_level', 'analysis_date']
                UIComponents.create_styled_dataframe(filtered_df[display_cols])
                
            else:
                st.info("No batch analysis results found.")
                
        except Exception as e:
            st.error(f"Error loading batch analysis data: {str(e)}")
    
    with tab3:
        display_combined_analytics()

def display_combined_analytics():
    """Combined analytics with charts and export"""
    st.subheader("Combined Analytics")
    
    try:
        # Load both databases
        single_results = pd.DataFrame(DatabaseManager.execute_query(
            SINGLE_ANALYSIS_DB, 'SELECT * FROM single_analysis_results'
        ))
        batch_results = pd.DataFrame(DatabaseManager.execute_query(
            BATCH_ANALYSIS_DB, 'SELECT * FROM batch_analysis_results'
        ))
        
        # Combine if data exists
        all_results = []
        if not single_results.empty:
            single_results['analysis_type'] = 'Single'
            all_results.append(single_results)
        if not batch_results.empty:
            batch_results['analysis_type'] = 'Batch'
            all_results.append(batch_results)
        
        if not all_results:
            st.info("No analysis data available.")
            return
        
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # Overall statistics
        metrics = {
            "Total Analyses": len(combined_results),
            "Glaucoma Cases": len(combined_results[combined_results['classification'] == 'glaucoma']),
            "High Risk": len(combined_results[combined_results['risk_level'] == 'High']),
            "Unique Patients": combined_results['patient_id'].nunique(),
            "Active Doctors": combined_results['doctor_id'].nunique()
        }
        UIComponents.display_metrics_row(metrics)
        
        # Time-based chart
        if len(combined_results) > 0:
            combined_results['analysis_date'] = pd.to_datetime(combined_results['analysis_date'])
            daily_counts = combined_results.groupby([
                combined_results['analysis_date'].dt.date,
                'classification'
            ]).size().reset_index(name='count')
            
            if not daily_counts.empty:
                fig = px.line(
                    daily_counts, 
                    x='analysis_date', 
                    y='count', 
                    color='classification',
                    title="Daily Analysis Trends",
                    color_discrete_map={'glaucoma': '#ff6b6b', 'non-glaucoma': '#66bb6a'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Export option
        csv_data = combined_results.to_csv(index=False)
        st.download_button(
            label="Download Combined Data (CSV)",
            data=csv_data,
            file_name=f"combined_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Error loading analytics data: {str(e)}")

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application controller"""
    # Initialize all databases
    DatabaseManager.init_all_databases()
    
    # Sidebar navigation
    st.sidebar.title("Glaucoma Analysis Dashboard")
    
    page = st.sidebar.selectbox(
        "Navigate to:",
        [
            "Doctor Portal",
            "Single Analysis", 
            "Batch Analysis",
            "Results Database",
            "About"
        ]
    )
    
    # Route to pages
    if page == "Doctor Portal":
        doctor_portal_page()
    elif page == "Single Analysis":
        single_analysis_page()
    elif page == "Batch Analysis":
        batch_analysis_page()
    elif page == "Results Database":
        results_database_page()
    elif page == "About":
        st.header("About Glaucoma Analysis Dashboard")
        st.markdown("""
        This dashboard provides AI-powered glaucoma risk analysis for healthcare professionals.
        
        **Key Features:**
        - Doctor registration with custom thresholds
        - Single and batch image analysis
        - Detailed contour visualizations showing:
          - Original fundus image
          - Predicted segmentation masks (OD/OC)
          - Vertical CDR analysis with measurement lines
          - Horizontal CDR analysis with measurement lines
          - Risk assessment and clinical metrics
        - Risk assessment and classification
        - Comprehensive results database (table format)
        - Export capabilities
        
        **Contour Visualization Features:**
        - Four-panel analysis display
        - Measurement lines showing diameter calculations
        - Color-coded contours (Red: Optic Disc, Blue: Optic Cup)
        - Real-time CDR calculations
        - Risk level assessment
        
        **Usage:**
        1. Register doctors in the Doctor Portal
        2. Perform single analysis with detailed contour visualization
        3. Use batch analysis for multiple images
        4. View results in tabular format in Results Database
        5. Export data for further analysis
        
        **Disclaimer:** This tool assists healthcare professionals and should not replace clinical judgment.
        """)

if __name__ == "__main__":
    main()