# Data Science Task 
### Author: Vincent Lunot
&nbsp;   
Information about the challenge is available in the file `challenge.md`.

## Overview

The task presented here is the study of the [PTB-XL dataset](https://physionet.org/content/ptb-xl/1.0.2/), a large publicly available electrocardiography dataset.  
All the computations and explanations are inside the Jupyter Notebook `notebook/ptb-xl_study.ipynb`.

The main steps of our study are:

- We first check the metadata to have an idea of what kind of information is provided.
- We next analyze the signal data, look for anomalies, and develop a simple algorithm to localize QRS-complexes.
- Finally, we show how to create and train a few simple neural networks on this data.

Note that while many information and tools are available online, we made the choice to study this dataset from scratch, developing ourselves our own simple algorithms. Using existing libraries whenever possible is usually the best practice in production, but our goal here is to get a better understanding of the dataset. That is why we decided to manipulate the data as much as possible.

## Instructions to run the notebook

### Download the data

Please go to the data directory and execute the script:
```bash
cd data
./download_data.sh
```

### Build and run Docker image

To be able to run the notebook in the same environment, you should first build a docker image:
```bash
docker build -t idoven-challenge:0.1 .
```

You can next run this docker image, sharing the notebook directory as well as the data directory (note that if your local 8888 port is already used, you can pick a new one, e.g. -p 9999:8888):
```bash
docker run -it --rm --gpus all -v [location of the notebook directory]:/tf/notebooks -v [location of the data directory]:/tf/data -p 8888:8888 idoven-challenge:0.1
```

Jupyter Lab is automatically started by the docker image, you can access it in your bowser by copying the address displayed after executing the previous command (note that if you have changed the port in the previous command, you should also change it here).

*You can now open the notebook inside Jupyter Lab, we hope you will enjoy it!*


## Summary of our work

### 1. Metadata

We restricted our study to a few columns of the metadata. There is a lot of information in the metadata, and asserting the quality of each field would take a lot of time. So we focused on the fields that seemed as the most important for a first study. 

### 2. ECG Signal

#### First check

We started by plotting a few ECG signals, to get a first idea of the data. We next looked for some signals with surprising values, and noticed that some of them have no annotation in the signal metadata section (no drift or noise or other problem annotated). We should therefore be very careful when using this signal metadata.

#### QRS-complexes

We next worked on developing a naive QRS-complex localization algorithm. Since the big peaks of each electrode signals are in similar locations, we thought about combining the signals of all the electrodes to make the QRS-complexes locations more visible. In order to combine each signal, we first centered them around zero by using the median values of sliding windows.

When testing our algorithm, we discovered that the R-peaks annotated in the metadata may not be very reliable. Indeed, we found R-peaks that were missing in the annotations.

#### Heart rate

We used our R-peaks computation to evaluate an instantaneous heart rate.

#### Repeated values in the signals

While testing the QRS-complex algorithm, we also noticed that a signal had a lot of repeating values (50) at its end. We checked if this happens for other records and discovered that most of the dataset has this problem (around 99.7% of the records!). Actually, all the records have the last 45 values of each signal that are repeated.

#### Q and S

We modified our algorithm to look for each of the Q, R and S points, rather than looking for the maximum combined amplitude. The S points seem easier to find than the Q points.  
We next tested the library NeuroKit2 to localize different waves of the ECG signal. We saw that the default method tends to miss peaks, further tests are necessary.
We also used NeuroKit2 to get statistics of the signal, including heart rate.

### 3. Deep learning experiments

In this section, we used only simple models that we can modify and train quickly. The goal was to assert quickly some possibilities of this data, and manipulate it a bit. The next step will be to read all the publications on that topic.

#### Data preparation

We first create the training, validation and test sets. We standardize the values per electrode signal.

#### Regression model

To get a first idea of the quality of the data, we first trained a simple regression model, once for predicting the age of the patient and another time for predicting the number of R-peaks in the signal. We saw that for each target, this simple model is able to provide non-trivial results. We therefore continued with other types of targets.

#### Segmentation model

We trained a simple segmentation model to predict the location of the R-peaks. The first results look promising.

#### Classification model

We trained a simple classification model to predict the superclass diagnostics. The first results look promising.

#### Data augmentation

We discussed briefly data augmentation. Having a good data augmentation pipeline could be a way to improve the results of complex models.

## Future work

- Read more information about ECG, diagnostics and devices: if we work in this area, it is important to understand perfectly well all the specifics.
- Do a full survey of published models for diagnostic prediction (and other topics of interest).
- Test SOTA models.
- Discuss on a regular basis a few selected signals with a cardiologist or any other expert in this area to get a better understanding of some of the problems.
- Work more on the dataset, look for other datasets to understand the differences, and prepare a data pipeline (with augmentation) for training new models.
- Study models for asserting data quality.
- Further test available toolboxes such as BioSPPy (https://biosppy.readthedocs.io) and NeuroKit2 (https://neuropsychology.github.io/NeuroKit/)
- Play with the repository Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL (https://github.com/helme/ecg_ptbxl_benchmarking)
- ...


## References

Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., & Schaeffter, T. (2022). PTB-XL, a large publicly available electrocardiography dataset (version 1.0.2). PhysioNet. https://doi.org/10.13026/zx4k-te85.

Wagner, P., Strodthoff, N., Bousseljot, R.-D., Kreiseler, D., Lunze, F.I., Samek, W., Schaeffter, T. (2020), PTB-XL: A Large Publicly Available ECG Dataset. Scientific Data. https://doi.org/10.1038/s41597-020-0495-6

Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220.

Python waveform-database (WFDB) package. https://wfdb.readthedocs.io/.

Heydarian et al., MLCM: Multi-Label Confusion Matrix, IEEE Access,2022. https://github.com/mrh110/mlcm

Pan, J. and Tompkins, W., 1985. A Real-Time QRS Detection Algorithm. IEEE Transactions on Biomedical Engineering, BME-32(3), pp.230-236.

Kramer Linus, Menon Carlo, Elgendi Mohamed (2022) ECGAssess: A Python-Based Toolbox to Assess ECG Lead Signal Quality, Frontiers in Digital Health, Volume 4.

Makowski, D., Pham, T., Lau, Z. J., Brammer, J. C., Lespinasse, F., Pham, H., Schölzel, C., & Chen, S. A. (2021). NeuroKit2: A Python toolbox for neurophysiological signal processing. Behavior Research Methods, 53(4), 1689-1696. https://doi.org/10.3758/s13428-020-01516-y

N. Strodthoff, P. Wagner, T. Schaeffter and W. Samek, "Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL," in IEEE Journal of Biomedical and Health Informatics, vol. 25, no. 5, pp. 1519-1528, May 2021, doi: 10.1109/JBHI.2020.3022989.
