## Data Science Task
This commit contains the submit from Cristina de la Torre for **Idoven** data scientist job offering.

The electrocardiogram (ECG) is a non-invasive representation of the electrical activity of the heart from electrodes placed on the surface of the torso. The standard 12-lead ECG has been widely used to diagnose a variety of cardiac abnormalities such as cardiac arrhythmias, and predicts cardiovascular morbidity and mortality. The early and correct diagnosis of cardiac abnormalities can increase the chances of successful treatments. However, manual interpretation of the electrocardiogram is time-consuming, and requires skilled personnel with a high degree of training.

Automatic detection and classification of cardiac abnormalities can assist physicians in the diagnosis of the growing number of ECGs recorded. Over the last decade, there have been increasing numbers of attempts to stimulate 12-lead ECG classification. Many of these algorithms seem to have the potential for accurate identification of cardiac abnormalities. However, most of these methods have only been tested or developed in single, small, or relatively homogeneous datasets. 

## **Achievements**

*   To know Physionet, and more specifically the PTB-XL Dataset. 
*   To know specific libraries related to Physionet to load, plot.. signals.
*   To investigate more about ECG and the relevant paper that it has in the heart issues detection.
*   To be more confident with the ECG area.
*   To apply the knowledge acquired during my master studies in the subject Biosignals and Machine Learning.
## **Comments**
During this task, I **has been able to** extract some features from ECG signal, that later will be relevant to detect some anomalies. Anomalies as for example Arrhythmia or any other heart disease. 

With respect to the **time needed** to develop this task. I want to comment that due to the computational cost of downloading all the data and process it, I finally decided to start with a simple task instead of a machine learning task as I wanted at first.

##  **Next steps**

*   **Test the code and the feature extraction in a large variety of ECG signals.**

Now due to the lack of time and the time consuming task, I was unable to test the code in the whole PTB-XL dataset.

*   **Introduce Machine Learning/Deep Learning in future steps:**

At first, introduce Machine Learning/Deep Learning to detect automatically the different peaks. In that way we will be able to later on use another Machine Learning/Deep Learning model to detect heart anomalies. 

But, how? This last model as first propposal I will do the following:

Feature Selection: First of all, I will see the correlation between all the features of each signal (age, heart beat, R peak distance...) and the disease/anomalie. In that way, I will be able to see which features are the most relevant in the anomalie detection and we can delete those features that can be adding noise to our model.

Train data, test data and validation data. After evaluating the data that PTB-XL is providing us we can see that there is patients with different anomalies. So, with the aim of obtaining a more accurate results. It is really important to have a balanced data in the model, in other words to have equitative the quantity of patients with each disease in the different sets of data.

Once all these steps have been carried out, we can start to try the different machine learning/deep learning models that exhist nowadays and are in the benchmark.
Thanks for giving this a go, we can't wait to see what you come up with.

# **References and Credits** 💳
Pan, J. and Tompkins, W., 1985. A Real-Time QRS Detection Algorithm. IEEE Transactions on Biomedical Engineering, BME-32(3), pp.230-236

Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., & Schaeffter, T. (2022). PTB-XL, a large publicly available electrocardiography dataset (version 1.0.3). PhysioNet. Link

Subject Biosignals and Bioimages. Master in Information Health Engineering (2019-2020).

Github repositories:
*   https://github.com/MIT-LCP/wfdb-python
*   https://github.com/adityatripathiiit/Pan_Tompkins_QRS_Detection
