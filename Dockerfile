# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster
EXPOSE 8888
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
ENTRYPOINT ["jupyter", "notebook", "analysis_notebook.ipynb", "--ip=\"0.0.0.0\"", "--allow-root"]
