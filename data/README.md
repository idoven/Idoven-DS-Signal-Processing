## Data Description
The ECGs in this collection were obtained using a non-commercial, PTB prototype recorder with the following specifications:

- 16 input channels, (14 for ECGs, 1 for respiration, 1 for line voltage)

- Input voltage: ±16 mV, compensated offset voltage up to ± 300 mV
- Input resistance: 100 Ω (DC)
- Resolution: 16 bit with 0.5 μV/LSB (2000 A/D units per mV)
 -Bandwidth: 0 - 1 kHz (synchronous sampling of all channels)
- Noise voltage: max. 10 μV (pp), respectively 3 μV (RMS) with input short circuit
- Online recording of skin resistance
- Noise level recording during signal collection

The database contains 549 records from 290 subjects (aged 17 to 87, mean 57.2; 209 men, mean age 55.5, and 81 women, mean age 61.6; ages were not recorded for 1 female and 14 male subjects). Each subject is represented by one to five records. There are no subjects numbered 124, 132, 134, or 161. Each record includes 15 simultaneously measured signals: the conventional 12 leads (i, ii, iii, avr, avl, avf, v1, v2, v3, v4, v5, v6) together with the 3 Frank lead ECGs (vx, vy, vz). Each signal is digitized at 1000 samples per second, with 16 bit resolution over a range of ± 16.384 mV. On special request to the contributors of the database, recordings may be available at sampling rates up to 10 KHz.

Within the header (.hea) file of most of these ECG records is a detailed clinical summary, including age, gender, diagnosis, and where applicable, data on medical history, medication and interventions, coronary artery pathology, ventriculography, echocardiography, and hemodynamics. The clinical summary is not available for 22 subjects. 

Access the files using the Google Cloud Storage Browser here. Login with a Google account is required.

Access the data using the Google Cloud command line tools (please refer to the gsutil documentation for guidance): 

    gsutil -m -u YOUR_PROJECT_ID cp -r gs://ptbdb-1.0.0.physionet.org DESTINATION

Download the files using your terminal: 

    wget -r -N -c -np https://physionet.org/files/ptbdb/1.0.0/


Download the zip file using your terminal: 

    wget https://physionet.org/static/published-projects/ptbdb/ptb-diagnostic-ecg-database-1.0.0.zip

