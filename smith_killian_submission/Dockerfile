####
#
# Docker configuration for Idoven job interview challenge.
#
####

# Use standard ubuntu machine
FROM ubuntu:22.04

# Update and install essential packages
RUN apt-get update
RUN apt-get install -y git wget

# Get conda and our required packages
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda
ENV PATH=$PATH:/miniconda/pcondabin:/miniconda/bin
COPY packages packages
RUN conda env update -f packages/environment.yml

# Setup primary environment (can add more environments)
RUN conda env create -f packages/environment_default.yml
SHELL ["conda","run","-n","default","/bin/bash","-c"]
RUN python -m ipykernel install --name kernel_default --display-name "Default"
RUN pip install -U -r packages/requirements_default.txt

# Cleanup after installing conda environments
RUN rm -rf packages
RUN rm Miniconda3-latest-Linux-x86_64.sh

# Setup docker user
ENV NB_USER guest
ENV NB_UID 1000
RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER

# Setup working directory
WORKDIR /home/${NB_USER}
USER $NB_USER

# Activate default environment
SHELL ["/bin/bash","-c"]
RUN conda init
RUN echo 'conda activate default' >> ~/.bashrc

# Copy local files to container
#RUN  mkdir -pv /etc/ipython/
#COPY  /notebook/data 
#COPY data /notebook/data 
#COPY README.md README.md
#RUN  mkdir -pv /etc/ipython/
#COPY ipython/ipython_config.py /etc/ipython/ipython_config.py

# Set jupyter port
# Note: ensure port is not blocked by firewall
EXPOSE 8888

# Default command for docker image
ENTRYPOINT [ "jupyter", "notebook", "--no-browser","--ip=0.0.0.0","--NotebookApp.token=''","--NotebookApp.password=''" ]

