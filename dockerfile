# Use a base image with Conda pre-installed
FROM continuumio/miniconda3:latest

# Set the working directory
WORKDIR /app

# Copy environment.yml file into the container
COPY environment.yml .

# Create the Conda environment
RUN conda env create -f environment.yml && conda clean -afy

# Install additional Python dependencies (including pytest)
RUN conda run -n basic_env pip install --no-cache-dir pytest

# Copy the rest of the project files
COPY . .

# Add /app to the Python path
ENV PYTHONPATH=/app

# Set the entry point for the container to use the Conda environment
ENTRYPOINT ["conda", "run", "-n", "basic_env"]

# Default command for the container
CMD ["python", "cli.py", "--root-dir", "data/archive/Rice_Image_Dataset", "--batch-size", "32", "--epochs", "10", "--lr", "0.001", "--save-path", "model_parameters/model.pth"]
