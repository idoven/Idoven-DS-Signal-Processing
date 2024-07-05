# Data Science Task
In this document I am going to explain about following.


1- Reference documentation used.

2- The work done and lessons learned.

3- Project files and How to run the implemented code.

4- Dataset

## References and the leearn lessons.
First I went through multiple videos from Youtube to learn about ECG. Three examples of references that I used to become familiare with ECG signals are as follows.

1- https://www.youtube.com/watch?v=u1m3HKW1VqU

2- https://en.wikipedia.org/wiki/Electrocardiography

3- https://en.wikipedia.org/wiki/QRS_complex

Using these references, I became familiar with different patterns that exist in ECG signals, such as the various intervals that are important to annotate, as well as the illustration of the QRS complex pattern. Additionally, using these resources, I noticed the appropriate visualizations that medical doctors may request.
 
To generate the visualizations, I took searches for existing signal processing packages. The implemented code is based on the package named "NeuroKit2". The repository of this package can be found in the address https://github.com/neuropsychology/NeuroKit. This package provides easy access to advanced biosignal processing routines. One of the reasons I used this package is that the majority of the signal processing community uses it. The repository at https://github.com/obss/BIOBSS provides a table of existing packages and highlights the features that each package has. Looking at the repository of NeuroKit2, we can see that it has 1.5k stars, which shows it is a commonly used package.



## Project files
This project includes following files.

1- SignalProcessing.ipynb: This notebook implements code for visualizing 12-lead ECG signals. It also illustrates heart rate, ECG signals and peaks, average heart shape, and average heart rate for each `ecg_id`. Additionally, the used packages annotate different characteristics of the signals, such as the R-peak. A detailed explanation of each code cell is provided in this notebook. You can change the values of some variables to visualize different signals from different patients. 

2- myplot.py. The functions `ecg_plot` and `signal_plot` from Neurokit2 were rendereing in wrong sizes. So, I reused these functions and made some minor modifications to resolve this issues. 

3- idoven_env.yml This file shows dependencies of the Anaconda environment for this project. For this project, I have created indivdual environment and installed the Neurokit2 in it. 

4- Docker and docker-compose files. These files include the command for installations such as anaconda env., Jupytherlab, as well as adding files. The provided docker-compose file defines a configuration for the Docker service. For example, it indicates the ports that Jupyter works on. In order to execute the code, the only thing that you need to do is open a terminal and insert one of the following commands depending on your docker version.

	- docker-compose up
	
	- docker compose up 


After opening the Jupyter lab in your browser, the kernel named 'idoven' should be already activated. If not, change the kernel of the jupyter from defult to the kernel named 'idoven'. 


5- ptbxl_database.csv: This is one of the metadata from the dataset. From this file, we can read specific ecg files (please see notebook for more details).

6- scp_statements.csv: We use the information of this file to add a column indicating the annotation/label of the signals. We read both signial and its corresponding annotation/label. 

 
## Dataset
Since the dataset is huge and takes a long time to download, I created a directory in the Docker file named "data". What you need to do is put the dataset in this directory. When you download the dataset, it has several subfolders. Please put the two subfolders named "records100" and "records500" in the data directory. As I mentioned earlier, to run the code you just need to insert one of the mentioned Docker commands in the internal.
 









