# Instructions
To reproduce the results there are three different ways how to run the notebook.

## Installation
Use the shell script `./download_download_data.sh` in the `./data/` folder to download the data if not already done.
The dataset should then be located in ./data/physionet.org/files/ptb-xl/1.0.2/ .
Make sure to navigate inside the folder 'Idoven-Data-Scientist' with 
`cd Idoven-Data-Scientist`. Then execute the following commands to run it:

### Docker
* `docker build --network=host -t ecg_analysis .`
* `docker run --rm -it -p 8888:8888 -p 5005:5005 -v ./data:/app/data ecg_analysis`

### Local pip environment
* `pip install -r requirements.txt`
* `jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser & mlflow ui -h 0.0.0.0 -p 5005t`

### Conda
* `conda env create -f environment.yaml`
* `conda activate ecg_analysis`
* `jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser & mlflow ui -h 0.0.0.0 -p 5005`

Use the URL in the console output to access the notebook with your browser 
(i.e. http://127.0.0.1:8888/lab?token=4608a0b0a472b4e9a32876bef7cd1107a4638c8b2ca6c5fc).
To see the experiments that were performed in mlflow, open http://127.0.0.1:5005.

## Folder Structure
The main notebook can be found in the file `analze_ecg_data.ipynb`.
The folder `/utils/` contains all files load load and process the ecg data.
The folder `/model/` contains all files of the deep learning model to classify the data.
The folder `/experiments/` contains configurations and the run file to run the model experiments.
