ARG BASE_CONTAINER=tverous/pytorch-notebook:latest
FROM $BASE_CONTAINER

ENV SHELL=/bin/bash

WORKDIR /app/

COPY requirements.txt .
COPY analyze_ecg_data.ipynb .
COPY train_model.py .
COPY model/ ./model/
COPY utils/ ./utils/
COPY mlruns/ ./mlruns/

USER root
RUN pip install -r requirements.txt
RUN pip list

RUN ls

#CMD echo ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"] &
#CMD echo ["mlflow", "ui"]
CMD ["/bin/bash", "-c", "jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser & mlflow ui -h 0.0.0.0 -p 5005"]

EXPOSE 8888
EXPOSE 5000