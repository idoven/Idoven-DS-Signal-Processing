ARG BASE_CONTAINER=pytorch/pytorch
FROM $BASE_CONTAINERLABEL

COPY environment.txt .
COPY data .
COPY analyze_ecg_data.ipynb .

USER rootRUN pip -e environment.txt
# Switch back to jovyan to avoid accidental container runs as root
USER $NB_UID