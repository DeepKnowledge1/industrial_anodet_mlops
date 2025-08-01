from locust import HttpUser, task
import base64
import io
from PIL import Image

class LoadTest(HttpUser):
    @task
    def test_inference(self):
        # Generate a dummy image in memory (RGB, 224x224)
        image = Image.new("RGB", (224, 224), color=(255, 0, 0))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        # Simulate sending the file as multipart/form-data
        files = {
            "file": ("dummy.png", buffer, "image/png")
        }

        # self.client.post("/predict", files=files)

        params = {"include_visualizations": False}  # Set to False for performance testing

        self.client.post("/predict", files=files, params=params)


#  locust -f load_testing\locustfile.py --host http://localhost:8000     