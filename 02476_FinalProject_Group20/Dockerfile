# UDKAST
# HUSK AT OPDATERE enviroment.yml og DOCKERFILE NÅR VI ER FÆRDIGE MED PROJEKTET

FROM python:3.10

# Working directory (vores er tom for nu)
WORKDIR /model

# system dependencies (opdater de her når vi er færdige med projektet)
# de her dependencies er bare pladsholdere
RUN apt-get update -y && apt-get install -y \
    libgl1-mesa-glx \
    ffmpeg \
    libsm6 \
    libxext6 \
    wget && \
    # fjern cache
    rm -rf /var/lib/apt/lists/*

# installer miniconda i vores docker
ENV CONDA_DIR /opt/conda

# download miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    # installer miniconda
    bash ~/miniconda.sh -b -p $CONDA_DIR && \
    # fjern miniconda.sh filen
    rm ~/miniconda.sh


# conda PATH
ENV PATH=$CONDA_DIR/bin:$PATH

# her kopieres alle filer fra repo til docker
COPY environment.yml environment.yml
COPY src/ src/
COPY models/ models/

# opsætning af environment og installering af dependencies
RUN conda env create -f environment.yml && \
    # cache fjernes
    conda clean -afy && \
    # aktiver environment
    echo "source activate docker_env" > ~/.bashrc

# Ensure the environment is used by adding it to PATH
ENV PATH=$CONDA_DIR/envs/docker_env/bin:$PATH


# Set working directory for the API
WORKDIR /model/src/api
ENV FLASK_APP=main
ENV FLASK_ENV=production

# Expose port and set command for the API
EXPOSE 8080
CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]
