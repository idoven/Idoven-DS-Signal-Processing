# Data Science Task
The purpose of this task is to extract useful data from the database given and identify the most important features of an ECG

## Setting environment
For the environment installation `pip install -r requirements.txt` command.

## Code usage
The functioning of the code is the following:

- The csv is converted into a dataframe where all the information can be consulted
- The ECG features are calculated in two ways, using hampel filter to detect P,R,S,T waves and using the library neurokit2. Both combined provide good results finding the waves.
- Filters have been passed to the signal in order to remove the baseline noise. When a higher freq bandpass filter was applied the signal were drastically reduced. (50-60Hz)
- The sampling frequency used was 500Hz.
- Measurements of amplitude and distance of the peaks were calculated between others.
- A simple knn algorithm was applied to estimate if a pacient suffer from ISCAL.
- The poblation used has 300 ECGs. 150 Normal and 150 ISCAL

## Results
Along the code the plots can be seen one by one and the smoothing methods. 
The results of the knn were not as expected, the variables extracted weren't significant for the final algorithm.

Dynamic Time Warping was an option to be implemented in order to compare each heartbeat.

Neurokit2 was found during the investigation about ECG signals and provide interesting graphs that compare the heartbeats and their waves like DTW method does.
