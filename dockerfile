# Use a base image with Conda pre-installed
FROM continuumio/miniconda3:latest

# Set the working directory
WORKDIR /app

# Copy environment.yml file into the container
COPY environment.yml .

# Create the Conda environment
RUN conda env create -f environment.yml && conda clean -afy

# Activate the environment and ensure it works (default name from environment.yml)
SHELL ["conda", "run", "-n", "basic_env", "/bin/bash", "-c"]

# Install additional dependencies (if needed)
RUN conda install -y pip && pip install --no-cache-dir typer

# Copy the rest of the project files
COPY . .

# Add /app to the Python path
ENV PYTHONPATH=/app

# Set the default command (activates the Conda environment)
CMD ["conda", "run", "-n", "basic_env", "python", "cli.py", "--root-dir", "data/archive/Rice_Image_Dataset", "--batch-size", "32", "--epochs", "10", "--lr", "0.001", "--save-path", "model_parameters/model.pth"]

