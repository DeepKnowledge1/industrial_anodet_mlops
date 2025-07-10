# prod_test.py
import os
import requests
import base64



def load_test_image(filename="007.png"):
    """Load test image and return base64 string"""
    image_path = os.path.join(os.path.dirname(__file__), filename)
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")



# -------------------------------
# Actual test function
# -------------------------------
def test_anomaly_service(score_uri, score_key, threshold, include_visualizations):
    """Run test against deployed anomaly detection endpoint"""

    assert score_uri, "Missing --score_uri"
    assert score_key, "Missing --score_key"

    image_base64 = load_test_image()
    input_data = {
        "image_base64": image_base64,
        "include_visualizations": include_visualizations,
        "threshold": threshold
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {score_key}"
    }

    response = requests.post(score_uri, headers=headers, json=input_data)

    assert response.status_code == 200, f"Status code: {response.status_code}"
    result = response.json()

    assert "anomaly_score" in result, "Missing anomaly_score"
    assert "is_anomaly" in result, "Missing is_anomaly"

    if include_visualizations:
        assert "boundary_image_base64" in result, "Missing boundary_image_base64"
        assert "heatmap_image_base64" in result, "Missing heatmap_image_base64"

    print(f"✅ Anomaly Score: {result['anomaly_score']} | Anomaly: {result['is_anomaly']}")
