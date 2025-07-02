
FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest

# System packages (optional, e.g., for OpenCV)
RUN apt-get update && apt-get install -y libgl1-mesa-glx 

# Copy and create the environment
COPY environment.yml .
RUN conda env create -f environment.yml

# Activate environment and set as default
SHELL ["conda", "run", "-n", "anomavision_env", "/bin/bash", "-c"]

# Alternative approach: Make conda environment the default
RUN echo "conda activate anomavision_env" >> ~/.bashrc
ENV PATH /opt/conda/envs/anomavision_env/bin:$PATH
ENV CONDA_DEFAULT_ENV=anomavision_env

# Ensure the environment is activated for any subsequent RUN commands
RUN conda init bash && \
    echo "conda activate anomavision_env" >> ~/.bashrc


