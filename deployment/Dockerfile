
FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest

# System packages (optional, e.g., for OpenCV)
RUN apt-get update && apt-get install -y libgl1-mesa-glx 

# Copy and create the environment
COPY environment.yml .
RUN conda env create -f environment.yml

# Activate environment and set as default
ENV PATH /opt/conda/envs/rg-ml-anomavision-dev/bin:$PATH
ENV CONDA_DEFAULT_ENV=rg-ml-anomavision-dev
