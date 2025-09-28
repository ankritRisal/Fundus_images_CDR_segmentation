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
from datetime import datetime
import os
import sqlite3
import zipfile
from pathlib import Path
import tempfile
import shutil
from typing import List, Dict, Any

# Import your modules (assuming they're in the same directory)
from app import UnetOutput
from models import UNet, load_model
from calculationTerms import CalculationUNET

# Configure the page
st.set_page_config(
    page_title="Advanced Glaucoma Detection System",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database initialization
DB_PATH = "glaucoma_database.db"

def init_database():
    """Initialize SQLite database for storing analysis results"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            patient_name TEXT,
            image_name TEXT NOT NULL,
            analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ai_vertical_cdr REAL,
            ai_horizontal_cdr REAL,
            doctor_vertical_cdr REAL,
            doctor_horizontal_cdr REAL,
            cdr_difference_v REAL,
            cdr_difference_h REAL,
            risk_level TEXT,
            od_height INTEGER,
            od_width INTEGER,
            oc_height INTEGER,
            oc_width INTEGER,
            doctor_notes TEXT,
            image_path TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def save_analysis_to_db(data: Dict[str, Any]):
    """Save analysis results to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO analysis_results 
        (patient_id, patient_name, image_name, ai_vertical_cdr, ai_horizontal_cdr,
         doctor_vertical_cdr, doctor_horizontal_cdr, cdr_difference_v, cdr_difference_h,
         risk_level, od_height, od_width, oc_height, oc_width, doctor_notes, image_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        data['patient_id'], data['patient_name'], data['image_name'],
        data['ai_vertical_cdr'], data['ai_horizontal_cdr'],
        data.get('doctor_vertical_cdr'), data.get('doctor_horizontal_cdr'),
        data.get('cdr_difference_v'), data.get('cdr_difference_h'),
        data['risk_level'], data['od_height'], data['od_width'],
        data['oc_height'], data['oc_width'], data.get('doctor_notes'),
        data.get('image_path')
    ))
    
    conn.commit()
    conn.close()

def get_all_results() -> pd.DataFrame:
    """Retrieve all analysis results from database"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('''
        SELECT * FROM analysis_results 
        ORDER BY analysis_date DESC
    ''', conn)
    conn.close()
    return df

def get_patient_results(patient_id: str) -> pd.DataFrame:
    """Retrieve results for a specific patient"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('''
        SELECT * FROM analysis_results 
        WHERE patient_id = ? 
        ORDER BY analysis_date DESC
    ''', conn, params=(patient_id,))
    conn.close()
    return df

# Custom CSS with enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 4px 8px rgba(255,107,107,0.3);
    }
    
    .risk-moderate {
        background: linear-gradient(135deg, #ffa726 0%, #fb8c00 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 4px 8px rgba(255,167,38,0.3);
    }
    
    .risk-low {
        background: linear-gradient(135deg, #66bb6a 0%, #43a047 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 4px 8px rgba(102,187,106,0.3);
    }
    
    .info-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-left: 4px solid #007bff;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .doctor-input-section {
        background: #fff3cd;
        border: 2px solid #ffeaa7;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .comparison-highlight {
        background: #e1f5fe;
        border-left: 4px solid #03a9f4;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
        border-bottom: 1px dotted #333;
    }
    
    .summary-stats {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 6px 12px rgba(116,185,255,0.3);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_glaucoma_model():
    """Load the trained U-Net model (cached for efficiency)"""
    try:
        device = torch.device("cpu")
        model = UNet(n_channels=3, n_classes=2)
        
        model_path = "/home/ankritrisal/Documents/project glaucoma /mainApp/binModel/best_seg.pth"
        if not os.path.exists(model_path):
            st.error(f"Model file '{model_path}' not found. Please ensure the model file is in the correct directory.")
            return None, None
            
        model = load_model(model, model_path, device)
        unet_op = UnetOutput(device)
        return model, unet_op
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def process_single_image(model, unet_op, image_path: str, patient_id: str, patient_name: str, doctor_vcdr: float = None, doctor_hcdr: float = None, doctor_notes: str = ""):
    """Process a single image and return results"""
    try:
        results = unet_op.predict_segmentation(model, image_path)
        
        # Calculate CDR differences if doctor input provided
        cdr_diff_v = None
        cdr_diff_h = None
        if doctor_vcdr is not None:
            cdr_diff_v = abs(results['pred_vCDR'] - doctor_vcdr)
        if doctor_hcdr is not None:
            cdr_diff_h = abs(results['pred_hCDR'] - doctor_hcdr)
        
        # Determine risk level
        max_cdr = max(results['pred_vCDR'], results['pred_hCDR'])
        if max_cdr > 0.6:
            risk_level = "High"
        elif max_cdr > 0.4:
            risk_level = "Moderate"
        else:
            risk_level = "Low"
        
        # Prepare data for database
        db_data = {
            'patient_id': patient_id,
            'patient_name': patient_name,
            'image_name': os.path.basename(image_path),
            'ai_vertical_cdr': results['pred_vCDR'],
            'ai_horizontal_cdr': results['pred_hCDR'],
            'doctor_vertical_cdr': doctor_vcdr,
            'doctor_horizontal_cdr': doctor_hcdr,
            'cdr_difference_v': cdr_diff_v,
            'cdr_difference_h': cdr_diff_h,
            'risk_level': risk_level,
            'od_height': results['od_height'],
            'od_width': results['od_width'],
            'oc_height': results['oc_height'],
            'oc_width': results['oc_width'],
            'doctor_notes': doctor_notes,
            'image_path': image_path
        }
        
        return results, db_data
        
    except Exception as e:
        st.error(f"Error processing {image_path}: {str(e)}")
        return None, None

def create_colored_dataframe(df: pd.DataFrame):
    """Create color-coded dataframe based on risk levels"""
    def highlight_risk(row):
        if row['risk_level'] == 'High':
            return ['background-color: #ffebee'] * len(row)
        elif row['risk_level'] == 'Moderate':
            return ['background-color: #fff3e0'] * len(row)
        elif row['risk_level'] == 'Low':
            return ['background-color: #e8f5e8'] * len(row)
        return [''] * len(row)
    
    return df.style.apply(highlight_risk, axis=1)

def create_risk_assessment_chart(vcdr, hcdr, doctor_vcdr=None, doctor_hcdr=None):
    """Enhanced CDR chart with doctor comparison"""
    fig = go.Figure()
    
    # Add risk zones
    fig.add_hrect(y0=0, y1=0.4, fillcolor="lightgreen", opacity=0.3, 
                  annotation_text="Low Risk", annotation_position="top left")
    fig.add_hrect(y0=0.4, y1=0.6, fillcolor="yellow", opacity=0.3,
                  annotation_text="Moderate Risk", annotation_position="top left")
    fig.add_hrect(y0=0.6, y1=1.0, fillcolor="lightcoral", opacity=0.3,
                  annotation_text="High Risk", annotation_position="top left")
    
    # AI CDR values
    fig.add_trace(go.Bar(
        x=['Vertical CDR', 'Horizontal CDR'],
        y=[vcdr, hcdr],
        name='AI Analysis',
        marker_color=['#1f77b4', '#ff7f0e'],
        text=[f'{vcdr:.3f}', f'{hcdr:.3f}'],
        textposition='auto',
    ))
    
    # Doctor CDR values if provided
    if doctor_vcdr is not None and doctor_hcdr is not None:
        fig.add_trace(go.Bar(
            x=['Vertical CDR', 'Horizontal CDR'],
            y=[doctor_vcdr, doctor_hcdr],
            name='Doctor Assessment',
            marker_color=['#d62728', '#ff9896'],
            text=[f'{doctor_vcdr:.3f}', f'{doctor_hcdr:.3f}'],
            textposition='auto',
            opacity=0.7
        ))
    
    fig.update_layout(
        title="Cup-to-Disc Ratio Analysis",
        yaxis_title="CDR Value",
        yaxis_range=[0, 1],
        barmode='group',
        height=500
    )
    
    return fig

def plot_enhanced_segmentation(results, use_container_width=True):
    """Create responsive segmentation visualization"""
    if results["input_image"].shape[0] == 3:
        display_image = results["input_image"].transpose(1, 2, 0)
    else:
        display_image = results["input_image"]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original image
    axes[0, 0].imshow(display_image)
    axes[0, 0].set_title('Original Fundus Image', fontsize=16, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Optic disc segmentation
    axes[0, 1].imshow(display_image, alpha=0.7)
    axes[0, 1].imshow(results["pred_od"], cmap='Reds', alpha=0.6)
    axes[0, 1].set_title('Optic Disc Segmentation', fontsize=16, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Optic cup segmentation
    axes[1, 0].imshow(display_image, alpha=0.7)
    axes[1, 0].imshow(results["pred_oc"], cmap='Blues', alpha=0.6)
    axes[1, 0].set_title('Optic Cup Segmentation', fontsize=16, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Combined visualization
    overlay = np.zeros((*display_image.shape[:2], 3), dtype=np.uint8)
    rim_only = results["pred_od"].astype(np.uint8) - results["pred_oc"].astype(np.uint8)
    overlay[rim_only > 0] = [255, 100, 100]
    overlay[results["pred_oc"] > 0] = [100, 100, 255]
    
    axes[1, 1].imshow(display_image, alpha=0.6)
    axes[1, 1].imshow(overlay, alpha=0.6)
    axes[1, 1].set_title('Combined Analysis\n(Red: Disc Rim, Blue: Cup)', fontsize=16, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    return fig

def main():
    # Initialize database
    init_database()
    
    # Header
    st.markdown('<h1 class="main-header">👁️ Advanced Glaucoma Detection System</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.header("🧭 Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "📊 Dashboard", 
        "🔍 Single Analysis", 
        "📁 Batch Analysis", 
        "📋 Results Database",
        "ℹ️ About"
    ])
    
    if page == "📊 Dashboard":
        dashboard_page()
    elif page == "🔍 Single Analysis":
        single_analysis_page()
    elif page == "📁 Batch Analysis":
        batch_analysis_page()
    elif page == "📋 Results Database":
        database_page()
    else:
        about_page()

def dashboard_page():
    """Dashboard with summary statistics"""
    st.header("📊 Clinical Dashboard")
    
    # Load all results
    results_df = get_all_results()
    
    if results_df.empty:
        st.info("📝 No analysis results found. Start by analyzing some images!")
        return
    
    # Summary statistics
    total_analyses = len(results_df)
    high_risk_count = len(results_df[results_df['risk_level'] == 'High'])
    moderate_risk_count = len(results_df[results_df['risk_level'] == 'Moderate'])
    low_risk_count = len(results_df[results_df['risk_level'] == 'Low'])
    unique_patients = results_df['patient_id'].nunique()
    
    # Summary cards
    st.markdown(f"""
    <div class="summary-stats">
        <h3>📈 Clinical Summary</h3>
        <div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
            <div><h4>{total_analyses}</h4><p>Total Analyses</p></div>
            <div><h4>{unique_patients}</h4><p>Unique Patients</p></div>
            <div><h4>{high_risk_count}</h4><p>High Risk Cases</p></div>
            <div><h4>{moderate_risk_count}</h4><p>Moderate Risk</p></div>
            <div><h4>{low_risk_count}</h4><p>Low Risk</p></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Risk distribution chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Risk Distribution")
        risk_counts = results_df['risk_level'].value_counts()
        fig_pie = px.pie(
            values=risk_counts.values, 
            names=risk_counts.index,
            color_discrete_map={
                'High': '#ff6b6b',
                'Moderate': '#ffa726', 
                'Low': '#66bb6a'
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("📅 Analysis Timeline")
        results_df['analysis_date'] = pd.to_datetime(results_df['analysis_date'])
        daily_counts = results_df.groupby(results_df['analysis_date'].dt.date).size().reset_index()
        daily_counts.columns = ['Date', 'Count']
        
        fig_line = px.line(daily_counts, x='Date', y='Count', title="Daily Analysis Count")
        st.plotly_chart(fig_line, use_container_width=True)
    
    # Recent high-risk cases
    high_risk_recent = results_df[results_df['risk_level'] == 'High'].head(5)
    if not high_risk_recent.empty:
        st.subheader("⚠️ Recent High-Risk Cases")
        display_cols = ['patient_id', 'patient_name', 'ai_vertical_cdr', 'ai_horizontal_cdr', 'analysis_date']
        st.dataframe(
            create_colored_dataframe(high_risk_recent[display_cols]), 
            use_container_width=True
        )

def single_analysis_page():
    """Single image analysis with doctor input"""
    st.header("🔍 Single Image Analysis")
    
    # Load model
    with st.spinner("Loading AI model..."):
        model, unet_op = load_glaucoma_model()
    
    if model is None or unet_op is None:
        st.error("Failed to load the model. Please check the model file and try again.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Patient information section
    st.subheader("👤 Patient Information")
    col1, col2 = st.columns(2)
    
    with col1:
        patient_id = st.text_input("Patient ID*", help="Unique identifier for the patient")
        patient_name = st.text_input("Patient Name", help="Patient's full name")
    
    with col2:
        doctor_notes = st.text_area("Doctor Notes", help="Additional clinical observations")
    
    if not patient_id:
        st.warning("Please enter a Patient ID to continue.")
        return
    
    # Image upload
    st.subheader("📷 Image Upload")
    uploaded_file = st.file_uploader(
        "Choose a fundus image...",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a retinal fundus image for glaucoma risk analysis"
    )
    
    if uploaded_file is not None:
        # Display uploaded image with responsive sizing
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            
        with col2:
            # Doctor input section
            st.markdown('<div class="doctor-input-section">', unsafe_allow_html=True)
            st.subheader("👨‍⚕️ Doctor Assessment")
            
            doctor_vcdr = st.number_input(
                "Doctor's Vertical CDR", 
                min_value=0.0, 
                max_value=1.0, 
                step=0.001, 
                format="%.3f",
                help="Doctor's assessment of vertical cup-to-disc ratio"
            )
            
            doctor_hcdr = st.number_input(
                "Doctor's Horizontal CDR", 
                min_value=0.0, 
                max_value=1.0, 
                step=0.001, 
                format="%.3f",
                help="Doctor's assessment of horizontal cup-to-disc ratio"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Analysis button
        if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
            with st.spinner("Analyzing image... This may take a few moments."):
                try:
                    # Save uploaded file temporarily
                    temp_path = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    image = Image.open(uploaded_file)
                    image.save(temp_path)
                    
                    # Process image
                    results, db_data = process_single_image(
                        model, unet_op, temp_path, patient_id, patient_name,
                        doctor_vcdr if doctor_vcdr > 0 else None,
                        doctor_hcdr if doctor_hcdr > 0 else None,
                        doctor_notes
                    )
                    
                    # Clean up temp file
                    os.remove(temp_path)
                    
                    if results and db_data:
                        # Save to database
                        save_analysis_to_db(db_data)
                        
                        # Display results
                        display_enhanced_results(results, db_data)
                        
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")

def batch_analysis_page():
    """Batch processing of multiple images"""
    st.header("📁 Batch Image Analysis")
    
    # Load model
    with st.spinner("Loading AI model..."):
        model, unet_op = load_glaucoma_model()
    
    if model is None or unet_op is None:
        st.error("Failed to load the model.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Batch settings
    st.subheader("⚙️ Batch Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        batch_patient_id = st.text_input("Base Patient ID", help="Base ID for batch processing")
        batch_patient_name = st.text_input("Patient Name", help="Name for all images in this batch")
    
    with col2:
        auto_increment = st.checkbox("Auto-increment Patient IDs", value=True, 
                                   help="Automatically add numbers to Patient ID for each image")
        batch_doctor_notes = st.text_area("Batch Notes", help="Notes applied to all images")
    
    # File upload options
    upload_option = st.radio("Upload Method:", ["Multiple Files", "ZIP Archive"])
    
    if upload_option == "Multiple Files":
        uploaded_files = st.file_uploader(
            "Choose multiple fundus images...",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Select multiple fundus images for batch analysis"
        )
        
        if uploaded_files and batch_patient_id:
            st.info(f"Selected {len(uploaded_files)} images for processing")
            
            if st.button("🚀 Process Batch", type="primary"):
                process_batch(uploaded_files, model, unet_op, batch_patient_id, 
                            batch_patient_name, auto_increment, batch_doctor_notes)
    
    else:  # ZIP Archive
        uploaded_zip = st.file_uploader(
            "Upload ZIP archive of images...",
            type=['zip'],
            help="Upload a ZIP file containing fundus images"
        )
        
        if uploaded_zip and batch_patient_id:
            if st.button("🚀 Process ZIP Archive", type="primary"):
                process_zip_batch(uploaded_zip, model, unet_op, batch_patient_id,
                                batch_patient_name, auto_increment, batch_doctor_notes)

def process_batch(uploaded_files, model, unet_op, base_patient_id, patient_name, auto_increment, notes):
    """Process multiple uploaded files"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()
    
    processed_count = 0
    total_files = len(uploaded_files)
    batch_results = []
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            status_text.text(f"Processing {uploaded_file.name} ({i+1}/{total_files})")
            
            # Determine patient ID
            if auto_increment:
                current_patient_id = f"{base_patient_id}_{i+1:03d}"
            else:
                current_patient_id = base_patient_id
            
            # Save temp file
            temp_path = f"temp_batch_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            image = Image.open(uploaded_file)
            image.save(temp_path)
            
            # Process image
            results, db_data = process_single_image(
                model, unet_op, temp_path, current_patient_id, 
                patient_name, None, None, notes
            )
            
            # Clean up temp file
            os.remove(temp_path)
            
            if results and db_data:
                # Save to database
                save_analysis_to_db(db_data)
                batch_results.append(db_data)
                processed_count += 1
            
            progress_bar.progress((i + 1) / total_files)
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    
    status_text.text(f"✅ Batch processing complete! Processed {processed_count}/{total_files} images.")
    
    # Display batch summary
    if batch_results:
        display_batch_summary(batch_results)

def process_zip_batch(uploaded_zip, model, unet_op, base_patient_id, patient_name, auto_increment, notes):
    """Process ZIP archive of images"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract ZIP
        with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Find image files
        image_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
        image_files = []
        
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if any(file.endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, file))
        
        if not image_files:
            st.error("No valid image files found in the ZIP archive.")
            return
        
        st.info(f"Found {len(image_files)} images in ZIP archive")
        
        # Process images
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        processed_count = 0
        batch_results = []
        
        for i, image_path in enumerate(image_files):
            try:
                status_text.text(f"Processing {os.path.basename(image_path)} ({i+1}/{len(image_files)})")
                
                # Determine patient ID
                if auto_increment:
                    current_patient_id = f"{base_patient_id}_{i+1:03d}"
                else:
                    current_patient_id = base_patient_id
                
                # Process image
                results, db_data = process_single_image(
                    model, unet_op, image_path, current_patient_id,
                    patient_name, None, None, notes
                )
                
                if results and db_data:
                    # Save to database
                    save_analysis_to_db(db_data)
                    batch_results.append(db_data)
                    processed_count += 1
                
                progress_bar.progress((i + 1) / len(image_files))
                
            except Exception as e:
                st.error(f"Error processing {os.path.basename(image_path)}: {str(e)}")
        
        status_text.text(f"✅ ZIP processing complete! Processed {processed_count}/{len(image_files)} images.")
        
        if batch_results:
            display_batch_summary(batch_results)

def display_batch_summary(batch_results):
    """Display summary of batch processing results"""
    st.subheader("📊 Batch Processing Summary")
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(batch_results)
    
    # Summary statistics
    risk_counts = summary_df['risk_level'].value_counts()
    total_processed = len(summary_df)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Processed", total_processed)
    with col2:
        st.metric("High Risk", risk_counts.get('High', 0))
    with col3:
        st.metric("Moderate Risk", risk_counts.get('Moderate', 0))
    with col4:
        st.metric("Low Risk", risk_counts.get('Low', 0))
    
    # Display results table
    display_cols = ['patient_id', 'patient_name', 'image_name', 'ai_vertical_cdr', 'ai_horizontal_cdr', 'risk_level']
    st.dataframe(create_colored_dataframe(summary_df[display_cols]), use_container_width=True)
    
    # Download batch results
    csv_data = summary_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Batch Results (CSV)",
        data=csv_data,
        file_name=f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def database_page():
    """Database view and management"""
    st.header("📋 Results Database")
    
    # Load all results
    results_df = get_all_results()
    
    if results_df.empty:
        st.info("📝 No analysis results found in database.")
        return
    
    # Database summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(results_df))
    with col2:
        st.metric("Unique Patients", results_df['patient_id'].nunique())
    with col3:
        latest_date = results_df['analysis_date'].max()
        st.metric("Latest Analysis", latest_date.split()[0] if latest_date else "N/A")
    
    # Filters
    st.subheader("🔍 Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_risk = st.multiselect(
            "Risk Level", 
            options=['Low', 'Moderate', 'High'],
            default=['Low', 'Moderate', 'High']
        )
    
    with col2:
        unique_patients = results_df['patient_id'].unique()
        selected_patients = st.multiselect(
            "Patient ID",
            options=unique_patients,
            default=[]
        )
    
    with col3:
        date_range = st.date_input(
            "Date Range",
            value=[],
            help="Select date range for filtering"
        )
    
    # Apply filters
    filtered_df = results_df.copy()
    
    if selected_risk:
        filtered_df = filtered_df[filtered_df['risk_level'].isin(selected_risk)]
    
    if selected_patients:
        filtered_df = filtered_df[filtered_df['patient_id'].isin(selected_patients)]
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df['date_only'] = pd.to_datetime(filtered_df['analysis_date']).dt.date
        filtered_df = filtered_df[
            (filtered_df['date_only'] >= start_date) & 
            (filtered_df['date_only'] <= end_date)
        ]
        filtered_df = filtered_df.drop('date_only', axis=1)
    
    st.info(f"Showing {len(filtered_df)} of {len(results_df)} records")
    
    # Display results table with color coding
    if not filtered_df.empty:
        # Select columns for display
        display_columns = [
            'id', 'patient_id', 'patient_name', 'image_name', 'analysis_date',
            'ai_vertical_cdr', 'ai_horizontal_cdr', 'doctor_vertical_cdr', 'doctor_horizontal_cdr',
            'cdr_difference_v', 'cdr_difference_h', 'risk_level', 'doctor_notes'
        ]
        
        display_df = filtered_df[display_columns].copy()
        
        # Format numerical columns
        for col in ['ai_vertical_cdr', 'ai_horizontal_cdr', 'doctor_vertical_cdr', 'doctor_horizontal_cdr', 'cdr_difference_v', 'cdr_difference_h']:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(3)
        
        # Display with color coding
        st.subheader("📊 Analysis Results")
        st.dataframe(create_colored_dataframe(display_df), use_container_width=True)
        
        # Patient-specific analysis
        st.subheader("👤 Patient-Specific Analysis")
        if selected_patients and len(selected_patients) == 1:
            patient_data = get_patient_results(selected_patients[0])
            display_patient_timeline(patient_data)
        
        # Download options
        st.subheader("💾 Download Options")
        col1, col2 = st.columns(2)
        
        with col1:
            # Download all filtered results
            csv_all = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Results",
                data=csv_all,
                file_name=f"filtered_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Download individual patient data
            if unique_patients.size > 0:
                selected_patient_download = st.selectbox(
                    "Select Patient for Individual Download",
                    options=unique_patients
                )
                
                if st.button("📥 Download Patient Data", use_container_width=True):
                    patient_data = get_patient_results(selected_patient_download)
                    csv_patient = patient_data.to_csv(index=False)
                    
                    st.download_button(
                        label=f"📥 Download {selected_patient_download} Data",
                        data=csv_patient,
                        file_name=f"patient_{selected_patient_download}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="patient_download"
                    )

def display_patient_timeline(patient_data):
    """Display timeline analysis for a specific patient"""
    if len(patient_data) > 1:
        st.subheader("📈 Patient CDR Timeline")
        
        # Prepare timeline data
        patient_data['analysis_date'] = pd.to_datetime(patient_data['analysis_date'])
        patient_data = patient_data.sort_values('analysis_date')
        
        # Create timeline chart
        fig = go.Figure()
        
        # AI CDR trends
        fig.add_trace(go.Scatter(
            x=patient_data['analysis_date'],
            y=patient_data['ai_vertical_cdr'],
            mode='lines+markers',
            name='AI Vertical CDR',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=patient_data['analysis_date'],
            y=patient_data['ai_horizontal_cdr'],
            mode='lines+markers',
            name='AI Horizontal CDR',
            line=dict(color='red')
        ))
        
        # Add doctor assessments if available
        doctor_v_data = patient_data.dropna(subset=['doctor_vertical_cdr'])
        doctor_h_data = patient_data.dropna(subset=['doctor_horizontal_cdr'])
        
        if not doctor_v_data.empty:
            fig.add_trace(go.Scatter(
                x=doctor_v_data['analysis_date'],
                y=doctor_v_data['doctor_vertical_cdr'],
                mode='markers',
                name='Doctor Vertical CDR',
                marker=dict(color='blue', symbol='diamond', size=10)
            ))
        
        if not doctor_h_data.empty:
            fig.add_trace(go.Scatter(
                x=doctor_h_data['analysis_date'],
                y=doctor_h_data['doctor_horizontal_cdr'],
                mode='markers',
                name='Doctor Horizontal CDR',
                marker=dict(color='red', symbol='diamond', size=10)
            ))
        
        # Add risk threshold lines
        fig.add_hline(y=0.4, line_dash="dash", line_color="orange", 
                      annotation_text="Moderate Risk Threshold")
        fig.add_hline(y=0.6, line_dash="dash", line_color="red", 
                      annotation_text="High Risk Threshold")
        
        fig.update_layout(
            title="Patient CDR Progression Over Time",
            xaxis_title="Analysis Date",
            yaxis_title="CDR Value",
            yaxis_range=[0, 1],
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # CDR change analysis
        if len(patient_data) > 1:
            latest_v = patient_data['ai_vertical_cdr'].iloc[-1]
            earliest_v = patient_data['ai_vertical_cdr'].iloc[0]
            v_change = latest_v - earliest_v
            
            latest_h = patient_data['ai_horizontal_cdr'].iloc[-1]
            earliest_h = patient_data['ai_horizontal_cdr'].iloc[0]
            h_change = latest_h - earliest_h
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Vertical CDR Change", 
                    f"{v_change:+.3f}",
                    delta=f"From {earliest_v:.3f} to {latest_v:.3f}"
                )
            with col2:
                st.metric(
                    "Horizontal CDR Change", 
                    f"{h_change:+.3f}",
                    delta=f"From {earliest_h:.3f} to {latest_h:.3f}"
                )
            
            if abs(v_change) > 0.05 or abs(h_change) > 0.05:
                st.warning("⚠️ Significant CDR change detected. Consider clinical review.")

def display_enhanced_results(results, db_data):
    """Display enhanced results with doctor comparison"""
    st.success("✅ Analysis completed!")
    
    # Extract values
    ai_vcdr = results['pred_vCDR']
    ai_hcdr = results['pred_hCDR']
    doctor_vcdr = db_data.get('doctor_vertical_cdr')
    doctor_hcdr = db_data.get('doctor_horizontal_cdr')
    risk_level = db_data['risk_level']
    
    # Risk assessment section
    st.subheader("🎯 Risk Assessment")
    risk_class = f"risk-{risk_level.lower()}"
    risk_messages = {
        'High': "⚠️ HIGH RISK: Immediate ophthalmologist consultation recommended",
        'Moderate': "⚠️ MODERATE RISK: Regular monitoring recommended", 
        'Low': "✅ LOW RISK: Normal optic disc appearance"
    }
    
    st.markdown(f'<div class="{risk_class}">{risk_messages[risk_level]}</div>', unsafe_allow_html=True)
    
    # Metrics comparison
    st.subheader("📊 CDR Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("AI Vertical CDR", f"{ai_vcdr:.3f}")
    with col2:
        st.metric("AI Horizontal CDR", f"{ai_hcdr:.3f}")
    with col3:
        if doctor_vcdr is not None:
            diff_v = abs(ai_vcdr - doctor_vcdr)
            st.metric("Doctor Vertical CDR", f"{doctor_vcdr:.3f}", f"Δ {diff_v:.3f}")
        else:
            st.metric("Doctor Vertical CDR", "Not provided")
    with col4:
        if doctor_hcdr is not None:
            diff_h = abs(ai_hcdr - doctor_hcdr)
            st.metric("Doctor Horizontal CDR", f"{doctor_hcdr:.3f}", f"Δ {diff_h:.3f}")
        else:
            st.metric("Doctor Horizontal CDR", "Not provided")
    
    # Comparison analysis
    if doctor_vcdr is not None or doctor_hcdr is not None:
        st.subheader("🔍 AI vs Doctor Comparison")
        
        if doctor_vcdr is not None:
            diff_v = abs(ai_vcdr - doctor_vcdr)
            agreement_v = "Good" if diff_v < 0.05 else "Moderate" if diff_v < 0.1 else "Poor"
            
            st.markdown(f"""
            <div class="comparison-highlight">
                <strong>Vertical CDR Agreement: {agreement_v}</strong><br>
                AI: {ai_vcdr:.3f} | Doctor: {doctor_vcdr:.3f} | Difference: {diff_v:.3f}
            </div>
            """, unsafe_allow_html=True)
        
        if doctor_hcdr is not None:
            diff_h = abs(ai_hcdr - doctor_hcdr)
            agreement_h = "Good" if diff_h < 0.05 else "Moderate" if diff_h < 0.1 else "Poor"
            
            st.markdown(f"""
            <div class="comparison-highlight">
                <strong>Horizontal CDR Agreement: {agreement_h}</strong><br>
                AI: {ai_hcdr:.3f} | Doctor: {doctor_hcdr:.3f} | Difference: {diff_h:.3f}
            </div>
            """, unsafe_allow_html=True)
    
    # Enhanced CDR chart
    st.subheader("📈 CDR Visualization")
    fig = create_risk_assessment_chart(ai_vcdr, ai_hcdr, doctor_vcdr, doctor_hcdr)
    st.plotly_chart(fig, use_container_width=True)
    
    # Segmentation visualization
    st.subheader("🔬 Segmentation Analysis")
    seg_fig = plot_enhanced_segmentation(results)
    st.pyplot(seg_fig, use_container_width=True)
    
    # Detailed measurements
    st.subheader("📏 Detailed Measurements")
    measurements_data = {
        'Measurement': [
            'Optic Disc Height (pixels)',
            'Optic Disc Width (pixels)', 
            'Optic Cup Height (pixels)',
            'Optic Cup Width (pixels)',
            'AI Vertical CDR',
            'AI Horizontal CDR'
        ],
        'Value': [
            results['od_height'],
            results['od_width'],
            results['oc_height'], 
            results['oc_width'],
            f"{ai_vcdr:.3f}",
            f"{ai_hcdr:.3f}"
        ]
    }
    
    if doctor_vcdr is not None or doctor_hcdr is not None:
        measurements_data['Doctor Assessment'] = [
            'N/A', 'N/A', 'N/A', 'N/A',
            f"{doctor_vcdr:.3f}" if doctor_vcdr is not None else 'N/A',
            f"{doctor_hcdr:.3f}" if doctor_hcdr is not None else 'N/A'
        ]
    
    measurements_df = pd.DataFrame(measurements_data)
    st.dataframe(measurements_df, use_container_width=True)
    
    # Download individual result
    st.subheader("💾 Download Results")
    
    # Create comprehensive results
    comprehensive_results = {
        "Patient ID": db_data['patient_id'],
        "Patient Name": db_data['patient_name'],
        "Image Name": db_data['image_name'],
        "Analysis Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "AI Vertical CDR": f"{ai_vcdr:.3f}",
        "AI Horizontal CDR": f"{ai_hcdr:.3f}",
        "Doctor Vertical CDR": f"{doctor_vcdr:.3f}" if doctor_vcdr else "Not provided",
        "Doctor Horizontal CDR": f"{doctor_hcdr:.3f}" if doctor_hcdr else "Not provided",
        "CDR Difference V": f"{db_data.get('cdr_difference_v', 0):.3f}" if db_data.get('cdr_difference_v') else "N/A",
        "CDR Difference H": f"{db_data.get('cdr_difference_h', 0):.3f}" if db_data.get('cdr_difference_h') else "N/A",
        "Risk Level": risk_level,
        "Optic Disc Height": results['od_height'],
        "Optic Disc Width": results['od_width'],
        "Optic Cup Height": results['oc_height'],
        "Optic Cup Width": results['oc_width'],
        "Doctor Notes": db_data.get('doctor_notes', '')
    }
    
    results_df = pd.DataFrame([comprehensive_results])
    csv = results_df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download Analysis Report (CSV)",
        data=csv,
        file_name=f"analysis_{db_data['patient_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    # Medical disclaimer
    st.warning("""
    **⚠️ Medical Disclaimer:** This tool is for research and educational purposes only. 
    It should not be used as a substitute for professional medical diagnosis or treatment. 
    Always consult with a qualified ophthalmologist for proper medical evaluation and care.
    """)

def about_page():
    """Enhanced about page"""
    st.header("ℹ️ About Advanced Glaucoma Detection System")
    
    # System overview
    st.subheader("🔬 System Overview")
    st.markdown("""
    This advanced glaucoma detection system combines state-of-the-art AI technology with 
    clinical workflow integration to provide comprehensive fundus image analysis for 
    glaucoma risk assessment.
    
    ### 🚀 Key Features
    
    **🔍 Analysis Capabilities:**
    - Single image analysis with doctor input comparison
    - Batch processing of multiple images or ZIP archives
    - Automated optic disc and cup segmentation using U-Net architecture
    - Real-time CDR calculation and risk assessment
    
    **👨‍⚕️ Clinical Workflow Integration:**
    - Patient information management with unique ID system
    - Doctor assessment input and AI vs. doctor comparison
    - Clinical notes and observations tracking
    - Color-coded risk level visualization
    
    **📊 Data Management:**
    - SQLite database for persistent storage
    - Comprehensive results tracking and history
    - Patient timeline analysis and progression monitoring
    - Flexible filtering and search capabilities
    
    **📱 User Experience:**
    - Responsive design for various screen sizes
    - Interactive charts and visualizations
    - Comprehensive download options (individual/batch/filtered)
    - Real-time progress tracking for batch operations
    """)
    
    # Technical details
    with st.expander("🔧 Technical Architecture"):
        st.markdown("""
        **AI Model:**
        - Architecture: U-Net with encoder-decoder structure
        - Input: RGB fundus images (auto-resized to 256x256)
        - Output: Binary segmentation masks for optic disc and cup
        - Training: Optimized for fundus image segmentation tasks
        
        **Database Schema:**
        - SQLite database for cross-platform compatibility
        - Comprehensive tracking of analysis parameters
        - Support for patient timeline and progression analysis
        - Efficient indexing for fast queries and filtering
        
        **Image Processing Pipeline:**
        1. Image upload and validation
        2. Preprocessing and normalization
        3. U-Net segmentation inference
        4. Post-processing and measurement extraction
        5. CDR calculation and risk assessment
        6. Results storage and visualization
        """)
    
    # Clinical interpretation
    with st.expander("🏥 Clinical Interpretation Guidelines"):
        st.markdown("""
        **Cup-to-Disc Ratio (CDR) Interpretation:**
        
        **📗 Normal Range (CDR < 0.4):**
        - Indicates healthy optic nerve appearance
        - Low risk for glaucomatous damage
        - Routine monitoring recommended
        
        **📙 Suspicious Range (CDR 0.4-0.6):**
        - May indicate early glaucomatous changes
        - Requires regular monitoring and clinical correlation
        - Consider additional testing (IOP, visual fields)
        
        **📕 Abnormal Range (CDR > 0.6):**
        - Strong indicator of glaucomatous optic nerve damage
        - High risk classification
        - Immediate ophthalmological evaluation recommended
        
        **🔍 AI vs Doctor Comparison:**
        - Difference < 0.05: Excellent agreement
        - Difference 0.05-0.10: Good agreement
        - Difference > 0.10: Requires clinical review
        
        **⚠️ Important Considerations:**
        - CDR values should be interpreted in clinical context
        - Consider patient history, family history, and other risk factors
        - AI results are supplementary to clinical judgment
        - Regular monitoring is essential for progression detection
        """)
    
    # Usage tips
    with st.expander("💡 Usage Tips & Best Practices"):
        st.markdown("""
        **📸 Image Quality Guidelines:**
        - Use well-focused, properly exposed fundus photographs
        - Ensure optic disc is clearly visible and centered
        - Avoid images with significant artifacts or reflections
        - Minimum recommended resolution: 256x256 pixels
        
        **👨‍⚕️ Clinical Workflow:**
        - Always enter patient ID for proper tracking
        - Use doctor assessment fields for comparison analysis
        - Add clinical notes for context and observations
        - Review AI results in conjunction with clinical findings
        
        **📊 Data Management:**
        - Use consistent patient ID formats for easy tracking
        - Utilize batch processing for efficiency
        - Regular database backups recommended
        - Filter results for focused analysis and reporting
        
        **🔍 Result Interpretation:**
        - Consider both vertical and horizontal CDR values
        - Monitor changes over time using patient timeline
        - Pay attention to AI vs doctor comparison metrics
        - Use color-coded risk levels for quick assessment
        """)
    
    # System limitations
    st.subheader("⚠️ System Limitations")
    st.warning("""
    **Important Limitations to Consider:**
    
    - This system is designed for research and educational purposes
    - AI results should supplement, not replace, clinical judgment
    - Image quality significantly affects analysis accuracy
    - System performance may vary with different fundus camera types
    - Results require validation by qualified medical professionals
    - Not intended for primary clinical diagnosis or treatment decisions
    
    **For Clinical Use:**
    Always consult with qualified ophthalmologists and follow established 
    clinical protocols for glaucoma diagnosis and management.
    """)
    
    # Version and credits
    st.subheader("📋 System Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Version:** 2.0.0
        **Last Updated:** 2024
        **Database:** SQLite
        **UI Framework:** Streamlit
        """)
    
    with col2:
        st.info("""
        **AI Model:** U-Net Architecture
        **Image Processing:** PIL, OpenCV
        **Visualization:** Plotly, Matplotlib
        **Data Analysis:** Pandas, NumPy
        """)

if __name__ == "__main__":
    main()