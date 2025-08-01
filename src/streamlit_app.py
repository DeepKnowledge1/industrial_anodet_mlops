import streamlit as st
import requests
from PIL import Image
import io
import base64
import json
import time
import sys
import argparse


# streamlit run src/streamlit_app.py -- --port 8080



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080, help="Port of the FastAPI server")
    parser.add_argument("--host", type=str, default="localhost", help="Host of the FastAPI server")
    return parser.parse_known_args()[0]



# Configuration
args = parse_args()
FASTAPI_URL = f"http://{args.host}:{args.port}"


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

# Custom CSS
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

st.markdown('<h1 class="main-header">🔍 Anomaly Detection System</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    # Health Check
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

    st.subheader("🖼️ Visualization Settings")
    include_visualizations = st.checkbox("Include Visualizations", value=True)

    st.subheader("🎯 Detection Threshold")
    new_threshold = st.slider(
        "Anomaly Threshold",
        min_value=0.1,
        max_value=50.0,
        value=float(current_threshold),
        step=0.1
    )

    st.subheader("📐 Resize Dimensions")
    resize_width = st.number_input("Resize Width", min_value=32, max_value=1024, value=900, step=8)
    resize_height = st.number_input("Resize Height", min_value=32, max_value=1024, value=900, step=8)

    if st.button("Update Params"):
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

# Main Layout
left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="Upload an image to analyze for anomalies"
    )
    if uploaded_file:
        st.image(uploaded_file, caption="📸 Original Image", use_container_width=True)

        # Run analysis directly after upload
        with st.spinner("🔄 Analyzing image..."):
            try:
                files = {
                    "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                }
                params = {"include_visualizations": include_visualizations}
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
                    st.session_state['analysis_result'] = result
                    st.session_state['analysis_time'] = analysis_time
                else:
                    st.error(f"❌ API Error {response.status_code}: {response.json().get('detail', 'Unknown error')}")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Connection Error: {str(e)}")
            except Exception as e:
                st.error(f"❌ Unexpected Error: {str(e)}")

with right_col:
    st.subheader("📊 Analysis Results")

    if 'analysis_result' in st.session_state:
        result = st.session_state['analysis_result']
        analysis_time = st.session_state.get('analysis_time', 0)
        anomaly_score = result['anomaly_score']
        is_anomaly = result['is_anomaly']

        r1, r2, r3 = st.columns(3)
        r1.metric("🎯 Anomaly Score", f"{anomaly_score:.3f}", f"Threshold: {new_threshold}")
        r2.metric("📋 Classification", "🔴 ANOMALY" if is_anomaly else "🟢 NORMAL")
        r3.metric("⏱️ Analysis Time", f"{analysis_time:.2f}s")

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

        # Visualizations
        if include_visualizations:
            def show_base64_image(base64_str, title):
                if base64_str:
                    try:
                        img_bytes = base64.b64decode(base64_str)
                        img = Image.open(io.BytesIO(img_bytes))
                        st.image(img, caption=title, use_container_width=True)
                    except Exception as e:
                        st.error(f"❌ Error displaying {title}: {str(e)}")
                else:
                    st.warning(f"⚠️ No data for {title}")

            st.subheader("🖼️ Visualizations")
            show_base64_image(result.get('heatmap_image_base64', ''), "🌡️ Heatmap Overlay")
            show_base64_image(result.get('boundary_image_base64', ''), "🖼️ Boundary Visualization")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>🔍 Anomaly Detection System | Powered by PaDiM</p>
    <p>Upload an image to detect anomalies using advanced machine learning</p>
</div>
""", unsafe_allow_html=True)
