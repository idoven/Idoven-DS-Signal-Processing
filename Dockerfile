# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster
EXPOSE 8888

WORKDIR /app
COPY . .

WORKDIR /app/data
RUN apt update
RUN apt -y install wget
RUN ./download_data.sh

WORKDIR /app
RUN pip install -r requirements.txt

ENTRYPOINT ["jupyter", "notebook", "--ip=\"0.0.0.0\"", "--allow-root"]
