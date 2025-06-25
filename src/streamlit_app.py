import streamlit as st
import requests
from PIL import Image
import io
import base64

FASTAPI_URL = "http://localhost:8000/predict"  # Update if deployed elsewhere

st.set_page_config(page_title="Anomaly Detection", layout="centered")
st.title("🔍 Anomaly Detection Interface")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button("Analyze Image"):
        # Send image to FastAPI
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(
            FASTAPI_URL,
            files={"file": ("image.png", uploaded_file.getvalue(), uploaded_file.type)},
            params={"include_visualizations": True}
        )

        if response.status_code == 200:
            result = response.json()

            st.subheader("📊 Results")
            st.markdown(f"**Anomaly Score:** {result['anomaly_score']:.2f}")
            st.markdown(f"**Is Anomaly:** {'✅ Yes' if result['is_anomaly'] else '❌ No'}")

            def show_base64_image(base64_str, title):
                if base64_str:
                    img_bytes = base64.b64decode(base64_str)
                    img = Image.open(io.BytesIO(img_bytes))
                    st.image(img, caption=title, use_container_width=True)

            st.subheader("🖼️ Visualizations")
            show_base64_image(result['anomaly_map_base64'], "Anomaly Map")
            show_base64_image(result.get('boundary_image_base64'), "Boundary Visualization")
            show_base64_image(result.get('heatmap_image_base64'), "Heatmap Visualization")

        else:
            st.error(f"API Error {response.status_code}: {response.json().get('detail', 'Unknown error')}")
