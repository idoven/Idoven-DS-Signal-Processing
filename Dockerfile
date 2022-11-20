# Based on tensorflow image
FROM tensorflow/tensorflow:2.10.0-gpu-jupyter

# Copy requirements.txt to our Docker image
COPY requirements.txt .
 
# Install requirements
RUN pip install --no-cache-dir -r requirements.txt

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter lab --ip 0.0.0.0 --no-browser --allow-root"]
