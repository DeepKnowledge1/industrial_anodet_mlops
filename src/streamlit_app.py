
import streamlit as st
import requests
from PIL import Image
import io
import base64
import json
import time

# Configuration
FASTAPI_URL = "http://localhost:8000"
PREDICT_ENDPOINT = f"{FASTAPI_URL}/predict"
HEALTH_ENDPOINT = f"{FASTAPI_URL}/health"
CONFIG_ENDPOINT = f"{FASTAPI_URL}/config"
MODEL_INFO_ENDPOINT = f"{FASTAPI_URL}/model-info"

st.set_page_config(
    page_title="Anomaly Detection System", 
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2e7bcf;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .anomaly-positive {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
    .anomaly-negative {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
    }
    .stImage > div {
        border: 2px solid #ddd;
        border-radius: 8px;
        padding: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Title and header
st.markdown('<h1 class="main-header">🔍 Anomaly Detection System</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar for configuration and status
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Health check
    try:
        health_response = requests.get(HEALTH_ENDPOINT, timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            status = health_data.get('status', 'unknown')
            model_type = health_data.get('model_type', 'unknown')
            current_threshold = health_data.get('threshold', 13.0)
            
            if status == 'healthy':
                st.success(f"✅ API Status: {status.upper()}")
                st.info(f"🤖 Model Type: {model_type.upper()}")
            else:
                st.error(f"❌ API Status: {status.upper()}")
        else:
            st.error("❌ API Connection Failed")
            current_threshold = 13.0
    except requests.exceptions.RequestException:
        st.error("❌ Cannot connect to API")
        current_threshold = 13.0
    
    st.markdown("---")
    
    # Threshold configuration
    st.subheader("🎯 Detection Threshold")
    new_threshold = st.slider(
        "Anomaly Threshold", 
        min_value=0.1, 
        max_value=50.0, 
        value=float(current_threshold), 
        step=0.1,
        help="Values >= threshold are classified as anomalies"
    )
    # Resize configuration
    st.subheader("📐 Resize Dimensions")
    resize_width = st.number_input("Resize Width", min_value=32, max_value=1024, value=900, step=8)
    resize_height = st.number_input("Resize Height", min_value=32, max_value=1024, value=900, step=8)
    
    if st.button("Update Threshold"):
        try:
            config_response = requests.post(
                CONFIG_ENDPOINT,
                json={
                    "threshold": new_threshold,
                    "resize_width": resize_width,
                    "resize_height": resize_height
                },
                timeout=5
            )
            if config_response.status_code == 200:
                st.success(f"✅ Config updated: Threshold = {new_threshold}, Resize = ({resize_width}, {resize_height})")
                st.rerun()
            else:
                st.error("❌ Failed to update config")
        except requests.exceptions.RequestException:
            st.error("❌ Cannot connect to API")
    
    st.markdown("---")
    
    # Model information
    if st.button("📊 Model Info"):
        try:
            model_response = requests.get(MODEL_INFO_ENDPOINT, timeout=5)
            if model_response.status_code == 200:
                model_data = model_response.json()
                st.json(model_data)
            else:
                st.error("❌ Failed to get model info")
        except requests.exceptions.RequestException:
            st.error("❌ Cannot connect to API")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="Upload an image to analyze for anomalies"
    )
    
    if uploaded_file:
        st.image(uploaded_file, caption="📸 Original Image", use_container_width=True)
        
        # Analysis options
        st.subheader("🔧 Analysis Options")
        include_visualizations = st.checkbox("Include Visualizations", value=True)
        
        if st.button("🚀 Analyze Image", type="primary"):
            with st.spinner("🔄 Analyzing image..."):
                try:
                    # Prepare the file for upload
                    files = {
                        "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                    }
                    params = {"include_visualizations": include_visualizations}
                    
                    # Send request to FastAPI
                    start_time = time.time()
                    response = requests.post(
                        PREDICT_ENDPOINT,
                        files=files,
                        params=params,
                        timeout=30
                    )
                    analysis_time = time.time() - start_time
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Store results in session state
                        st.session_state['analysis_result'] = result
                        st.session_state['analysis_time'] = analysis_time
                        st.success(f"✅ Analysis completed in {analysis_time:.2f}s")
                        
                    else:
                        error_detail = response.json().get('detail', 'Unknown error')
                        st.error(f"❌ API Error {response.status_code}: {error_detail}")
                        
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Connection Error: {str(e)}")
                except Exception as e:
                    st.error(f"❌ Unexpected Error: {str(e)}")

with col2:
    st.subheader("📊 Analysis Results")
    
    if 'analysis_result' in st.session_state:
        result = st.session_state['analysis_result']
        analysis_time = st.session_state.get('analysis_time', 0)
        
        # Results summary
        anomaly_score = result['anomaly_score']
        is_anomaly = result['is_anomaly']
        
        # Metrics display
        col2_1, col2_2, col2_3 = st.columns(3)
        
        with col2_1:
            st.metric(
                label="🎯 Anomaly Score",
                value=f"{anomaly_score:.3f}",
                delta=f"Threshold: {new_threshold}"
            )
        
        with col2_2:
            status_text = "🔴 ANOMALY" if is_anomaly else "🟢 NORMAL"
            st.metric(
                label="📋 Classification",
                value=status_text
            )
        
        with col2_3:
            st.metric(
                label="⏱️ Analysis Time",
                value=f"{analysis_time:.2f}s"
            )
        
        # Classification indicator
        if is_anomaly:
            st.markdown(f"""
            <div class="metric-container anomaly-positive">
                <h4>🚨 ANOMALY DETECTED</h4>
                <p>Score: {anomaly_score:.3f} >= {new_threshold} (threshold)</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-container anomaly-negative">
                <h4>✅ NORMAL IMAGE</h4>
                <p>Score: {anomaly_score:.3f} < {new_threshold} (threshold)</p>
            </div>
            """, unsafe_allow_html=True)
        
    else:
        st.info("👆 Upload an image and click 'Analyze Image' to see results")

# Visualizations section
if 'analysis_result' in st.session_state and include_visualizations:
    result = st.session_state['analysis_result']
    
    st.markdown("---")
    st.subheader("🖼️ Visualizations")
    
    def show_base64_image(base64_str, title, help_text=""):
        if base64_str:
            try:
                img_bytes = base64.b64decode(base64_str)
                img = Image.open(io.BytesIO(img_bytes))
                # Removed the 'help' argument from st.image()
                st.image(img, caption=title, use_container_width=True) 
                return True
            except Exception as e:
                st.error(f"❌ Error displaying {title}: {str(e)}")
                return False
        else:
            st.warning(f"⚠️ No data available for {title}")
            return False
    
    # Create columns for visualizations
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        # show_base64_image(
        #     result.get('anomaly_map_base64', ''), 
        #     "🗺️ Anomaly Map",
        #     "Heat map showing anomaly intensity across the image"
        # )
        
        show_base64_image(
            result.get('heatmap_image_base64', ''), 
            "🌡️ Heatmap Overlay",
            "Original image with anomaly heatmap overlay"
        )
    
    with viz_col2:
        show_base64_image(
            result.get('boundary_image_base64', ''), 
            "🖼️ Boundary Visualization",
            "Image with anomaly boundaries highlighted"
        )
        
        # show_base64_image(
        #     result.get('highlighted_image_base64', ''), 
        #     "✨ Highlighted Anomalies",
        #     "Image with anomalous regions highlighted in purple"
        # )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>🔍 Anomaly Detection System | Powered by PaDiM</p>
    <p>Upload an image to detect anomalies using advanced machine learning</p>
</div>
""", unsafe_allow_html=True)
