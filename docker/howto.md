<!-- 

# Build the Docker image from the project root
# docker build -f docker/Dockerfile -t fastapi-anomavision:latest .



# Run the container with model files mounted as volumes
# This allows you to keep model files on the host and not in the image

# docker run -d `
#   --name fastapi-app `
#   -p 8000:8000 `
#   --restart unless-stopped `
#   -v $(pwd)/padim_model.onnx:/app/padim_model.onnx `
#   -v $(pwd)/distributions:/app/distributions `
#   fastapi-anomavision:latest

# Alternative: Run without volume mounts (if models are copied into image)
docker run -d `
  --name fastapi-app `
  -p 8000:8000 `
  --restart unless-stopped `
  fastapi-anomavision:latest

# View logs
# docker logs fastapi-app

# Follow logs in real-time
# docker logs -f fastapi-app

# Test the health endpoint
# curl http://localhost:8000/health

# Test the API
# curl http://localhost:8000/

# Stop and remove container
# docker stop fastapi-app && docker rm fastapi-app

# For development with volume mounting (to see code changes without rebuild)
# docker run -d `
#   --name fastapi-app-dev `
#   -p 8000:8000 `
#   -v $(pwd)/src/fastapi_app.py:/app/fastapi_app.py `
#   -v $(pwd)/src/AnomaVision:/app/AnomaVision `
#   -v $(pwd)/padim_model.onnx:/app/padim_model.onnx `
#   -v $(pwd)/distributions:/app/distributions `
#   fastapi-anomavision:latest

# Check container size
# docker images fastapi-anomavision:latest

# Clean up unused images
# docker image prune -f -->