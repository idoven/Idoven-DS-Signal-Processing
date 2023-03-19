# Data Science Task
This page has details of how I approached the task assigned by **Idoven** for the data scientist job offering.

## Contact info

- Do not hesitate to send me an mail = [Jaime Cortón González](mailto:corton35jaime@corton35jaime?subject=[IDOVEN]%20Comment%20from%20Idoven%20Challenge)
- Find me on LinkedIn = [Jaime's LinkedIn](https://www.linkedin.com/in/jaime-crtn-gnzlz/)

## What I had to do

We ask data scientist do want to join **Idoven** to work with anonymised patient data, and on the basis of this data be able to:
- Be able to read the _ECG_ files and corresponding annotations
- Show how they will work on the signal and plot the signal in appropriate manner to be read by a doctor
- Identify the heart beat of the signal, average and total heart beat in the signal
- Identify the complex QRS in the signal and been able to annotate on it

As a result we expect a github project with and extructure that will include:
- Reference documentation used
- Jupyter Notebook, in an running environment, Colab, Docker.
- An explanation of the work done and lessons learned.


## Time schedule

**DISCLAIMER:** This practical interview/challenge was done after my 8 hours/day shift during one of the biggest transportation strikes in Paris, spending 1:30h to go to the office and 1:30 to go back home. Also it is important to say that on the weekends I am mainly writting my Master thesis since at work I am not allowed to do that (writting the document itself). I don't want to use this as any kind of excuse, but I think it is good to explain the context in which I developed the code.

__Monday__: Forked the project and started locally a folder for doing version control. Setting virtual environment on Visual Studio Code. 45 MIN

__Tuesday__: Load Database from csv, familiarization with WFDB, Trying first plottings analysis of the annotations. 2 HOURS

__Wednesday__: Basic structure construction and "development plan" organization. Documentation about useful libraries. 1 HOUR

__Thursday__: NO TIME TO WORK

__Friday__: 12-lead plotting + QRS detection + Heartbeat computation functions development. 1 HOURS 30 MIN

__Saturday__: Finishing friday's functions. Coding of the widgets and improval of the user experience for non coding users, improve of the visualizations and comment curation. 5 HOURS

__Sunday__: Final arrangement, Readme.MD completion, final commit. 5 HOURS

TOTAL WORK: around 15 HOURS 15 MINUTES 

##  Introduction

### 12 lead ECG Context

In this database we are dealing with 12-lead ECG signals. As the name indicates this ECG is conformed of 12 different signals thet capture specific electric pulses between different strategically selected parts:

* **BIPOLAR LIMB LEADS:**
    * **Lead I)** records the electrical activity between the right arm (-) and left arm (+)

    * **Lead II)** records the electrical activity between the right arm (-) and left leg (+)

    * **Lead III)** records the electrical activity between the left arm (-) and left leg (+)


* **UNIPOLAR LIMB LEADS:**
    * **aVR)** records the electrical activity of the heart from a perspective looking towards the right arm

    * **aVL)** records the electrical activity of the heart from a perspective looking towards the left arm

    * **aVF)** records the electrical activity of the heart from a perspective looking towards the left leg
    

* **(PRECORDIAL) UNIPOLAR CHEST LEADS:**
    * **V1)** records the electrical activity of the heart from the fourth intercostals space, right sterna border.

    * **V2)** records the electrical activity of the heart from the fourth intercostals space, left sterna border.

    * **V3)** records the electrical activity of the heart from a position between V2 and V4.

    * **V4)** records the electrical activity of the heart from a position in the left mid-clavicular line.

    * **V5)** records the electrical activity of the heart from the left anterior axillary line.

    * **V6)** records the electrical activity of the heart from the left mid-axillary line.

The 12-lead ECG is used to detect various heart conditions, such as arrhythmias, ischemia, and infarction. It provides a comprehensive view of the heart's electrical activity, allowing healthcare providers to diagnose and treat heart conditions accurately.

![image.png](attachment:image.png)
**Figure 1.** 12-Lead ECG Illustration

### A priori data description:

We have a dataset that consists on 2 main folders (**records100** and **records500**) containing the low resolution or high resolution files depending on the sampling frequency (100 for the ls and 500 for the hs). 

Appart from that, we also have 2 .csv files, one containing general information of each ecg like the patient_id, age, sex, nurse id (**ptbxl_database.csv**) and the other containing the list of diagnosis with its corresponding code and details (**scp_statements.csv**).

There is also a .py file with a dummmy example on how to charge the dataset (**example_physionet.py**). The rest of the files are not relevant for our study case and are mainly Licenses, formalities and metadata. A more detailed explanation of the data will be done below.

These annotations contains the classification of the ECG into 5 Superclases (A single patient/ecg can present several conditions at the same time). According to the official documentation (https://physionet.org/content/ptb-xl/1.0.2/), the 5 classes are the following:

1. **[NORM]**: Normal ECG

2. **[MI]**: Myocardial Infarction

3. **[STTC]**: ST/T Change

4. **[CD]**: Conduction Disturbance

5. **[HYP]**: Hypertrophy

##  Learned lessons:

### In general

1. It is always important to know the data that you are working with, before jumping to play it is always good to have a deep initial analysis using plots and visual representations to have some initial insights about the problem that you are dealing with.

2. Scheduling is always important, you have a task to accomplish and the first thing to do is to divide the goal into smaller steps to have a smooth path to follow towards that goal.

3. Interpretation of an ecg can be hard if you  don't the proper representation of the signal, so make sure you use the proper libraries to detect correctly all the important peaks and waves to facilitate clinicians their diagnosis and interpretation job.

### In detail

It is interesting to see how the leads and its strategically placed position can give us valuable clues to detect certain conditions, having an abnormal ST-T segment/wave (remember that this wave is almost flat in normal scenarios 'the plateau phase', in which the majority of the myocardial cells had gone through depolarization but not repolarization) can come from very different scenarios. Either an ischemia is causing an abnormal repolarization that changes the ST-T wave or a bundle branch block is causing an abnormal depolarization that will cause at the same time an abnormal repolarization that will generate an abnormal ST-T wave. I like to see the ecg analysis as some short of chain reaction in which once you find something in the data (let's call it a link of the chain) you have to follow it through the path discarding the non suitable options.

Another interesting detail is the non-ecg related data, it is always important to read the documentation to discard data that might add biases to your models (for instance there are some age values that are set to 300 because some issue of data protection of the elder people that if you don't notice will add incredible outliers to the dataset).

It is also key to know the relationship between the signals components, for example, if we have an abnormal/altered QRS complex due to an abnormal depolarization it is very likely that we will see an altered ST-T segment because of the abnormal repolarization caused by the previous abnormal depolarization, in the end diagnosis is searching the cause of abnormalities (in this case abnormalities of the ecg signal), so to avoid misleading conclussions it is important to know the conexions between the fragments.

### Extra (What I would do with more time)

- First of all, I would like to introduce an idea that I would like to comment with you. It is very common to analyze ECGs with neural networks based on temporal/sequential information like RNNs, LSTMs, transformer and many other state-of-the-art architectures but I've never seen somebody using computer vision models. Considering that the way a doctor interprets the ecg is through the eyes and through visual information of the signal, it is strange to me that the research is not directed into this area more often. It is true that analyzing the discrete ecg signal as a temporal series has a lot of advantages like the precision on voltage values analysis, the meaningful Fourier domain analysis possibility and many other things, but I think that having the computer vision models of nowadays joined with the text-to-image models there is a big path for research (It also opens the possibility of combining ecg signal with Cardiac MRI analysis and its relationship). It might be a good idea to find a common ground between ecg visual analysis together with the written clinical history.

- Having more time I would have make good use of the already done training, test and validation data partition. It was clear to me that you were expecting some algorithm development using ecg's but as stayed on the DISCLAIMER section, I didn't have much time. Ideally I would have set CUDA usage (typically on torch since CUDA 12 for tensorflow/keras is giving problems due to unreleased sublibraries) to run things on my personal GPU. I would have generated a better user interface to make doctors interpretations simpler considering the no-coding typical background, something simple like tkinter would be okey for the moment. I think I didn't update GitHub as often as I should have done, but again, time constraints.

- Also, having more time I would have documented everything better, I would have followed more in detail PEP 8 style guide to have a more readable code. I would have read more documentation regarding not only ecg fundamentals topics but also regarding python libraries and functions like wfdb (I think the library has a huge potential, also neurokit2 library)

- Finally I would like to say that you can contact me to have any kind of discussion, I'll be happy to set a meeting. Please feel free to make any kind of comment regarding the code, or the README.md file. Thank you for the opportunity.


Jaime Cortón González