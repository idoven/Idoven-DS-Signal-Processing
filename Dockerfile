# syntax=docker/dockerfile:1

FROM ubuntu:latest
WORKDIR /opt
COPY requirements.txt requirements.txt
COPY Code.ipynb Code.ipynb
RUN apt update -y
RUN apt install wget software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install python3.10 -y
RUN apt install python3-pip -y
RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt
RUN wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.2/
CMD ["python3", "-m", "notebook"]
EXPOSE 8888
