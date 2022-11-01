# Dataset
It's a 12-lead (I, II, III, aVL, aVR, aVF, V1–V6) ECG-waveform dataset comprising 21837 records from 18885 patients of 10 seconds length. It's assumed at rest. 
Although the original dataset has a sampling frequency of 400 Hz, it's been used the downsampled version at a sampling frequency of 100 Hz (100 samples per second). 
Seeing as a numpy array, each patient has a shape of 1000 × 12 values. Therefore, it's a multi-label dataset preceded by 71 unique SCP-ECG statements as classes performed by up to two cardiologists. Although the original dataset can contain up to three types of statements (diagnostic, form and rhythm), I have used the 5 diagnostic superclasses as diagnostic labels: NORM (Normal ECG), MI (Myocardial Infarction), STTC (ST/T Change), CD (Conduction Disturbance) and HYP (Hypertrophy).

Click [here](https://physionet.org/content/ptb-xl/1.0.2/) to know more about the dataset PTB-XL (a large publicly available electrocardiography dataset).

# ECG interpretation
An electrocardiogram (ECG) is a non-invasive representation of the electrical activity of the heart from electrodes placed on the surface of the torso. 
An ECG provides information about the heart rate and rhythm but its interpretation, the ability to determine whether the ECG waves and intervals (P-waves, PR-segments, QRS-complex, ST-segments, T-waves, PP-intervals, RR-intervals, TT-intervals, and many more) don't work properly, is time-consuming, and requires skilled personnel with a high degree of training. For instance, a myocardial infarction can be seen when there is a rise in the ST-segment, changes in the shape or flipping of T-waves, new Q-waves, or a new left bundle branch block, in any of the 12-leads.

Click [here](https://en.wikipedia.org/wiki/Electrocardiography) to know more about ECG.

# Data science tasks
The next 'What to do' data science tasks have been achieved after looking the dataset:

* heart_rate_variability.ipynb: It's a jupyter notebook running in google colab to build a function able to print the heart rate and the RR-interval variability from a 12-lead ECG-waveform. Click [here](https://en.wikipedia.org/wiki/Heart_rate_variability) to know more about heart rate variability. 

* diagnostic_superclasses_statistics.ipynb: It's a jupyter notebook running on google colab to do some statistics identifying whether a normal 12-lead ECG waveform can be differentiated (a distinctly different pattern) from all other diagnostic superclasses MI, STTC, CD, and HYP.

* anomaly_detection.ipynb: It's a jupyter notebook running in google colab to build a function able to alert whether a 12-lead ECG-waveform is deviated from normal behavior by using a LSTM (Long short-term memory) Autoencoder. Click [here](https://en.wikipedia.org/wiki/Long_short-term_memory) to know more about LSTM.

## Comments
* download_data.ipynb is a jupyter notebook to download the files of PTB-XL
* utils.py is a helper python script listing some functions used across the jupyter notebooks
* requirements.txt is a text file listing the modules and packages required by the jupyter notebooks. It can be generated by the jupyter notebook requirements.ipynb
* checkpoint.pt is a file where the state of the model is shown in case of restarting the training process.
* The helper function load_ECGs loads a .npy file (or generates it if file doesn't exist listing the physical signals of some ECGs into the folder records100). This folder splits the ECGs into training, validation and test sets for the jupyter notebook anomaly_detection.ipynb
