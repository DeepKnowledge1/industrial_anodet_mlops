FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest

# System packages
RUN apt-get update && apt-get install -y libgl1-mesa-glx && apt-get clean

# Copy environment file
COPY environment.yml .
# Create conda environment
RUN conda env create -f environment.yml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "anomavision_env", "/bin/bash", "-c"]

# Verify the environment works
RUN python -c "import numpy; print('NumPy version:', numpy.__version__)"
RUN python -c "import torch; print('PyTorch version:', torch.__version__)"

# Set the default environment
ENV CONDA_DEFAULT_ENV=anomavision_env
ENV PATH /opt/conda/envs/anomavision_env/bin:$PATH

# Ensure conda is initialized and environment is activated by default
RUN conda init bash
RUN echo "conda activate anomavision_env" >> ~/.bashrc