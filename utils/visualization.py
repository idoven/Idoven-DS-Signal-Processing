import pandas as pd
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import wfdb
import wfdb.processing
import ecg_plot
from scipy import signal

def barplot_cxs(d:pd.DataFrame):
    z1=pd.crosstab(d.diagnostic_superclass, d.sex_cat)\
        .apply(lambda r: 100*r/r.sum(), axis=1).round(2)\
        .stack().rename("Frequency [%]").reset_index()
    z2=pd.crosstab(d.diagnostic_binary_superclass, d.sex_cat)\
            .apply(lambda r: 100*r/r.sum(), axis=1).round(2)\
            .stack().rename("Frequency [%]").reset_index()
    
    with plt.style.context("seaborn-poster"):
        fig, ax = plt.subplots(1,2,figsize=(20,10))
        sns.set_palette("muted")
        fig.suptitle("Frequency [%] of Diagnostic Superclasses per Sex", fontsize=20, fontweight="bold")
        ax1, ax2 = ax.flatten()

        fig.patch.set_facecolor("w")
        ax1.set_facecolor("whitesmoke")
        ax1.set_title("Multiclass Version", fontsize=18)
        sns.barplot(data=z1, x="diagnostic_superclass", y="Frequency [%]", hue="sex_cat", ax=ax1)
        ax1.set_xlabel("Diagnostic Superclasses")
        ax1.set_ylabel("Frequency [%]")

        ax2.set_facecolor("whitesmoke")
        ax2.set_title("Binary Version", fontsize=18)
        sns.barplot(data=z2, x="diagnostic_binary_superclass", y="Frequency [%]", hue="sex_cat", ax=ax2)
        ax2.set_xlabel("Diagnostic Superclasses")
        ax2.set_ylabel("Frequency [%]")
        plt.show()

        
def hist_age(df:pd.DataFrame):
    d = df.copy()
    d.reset_index(inplace=True)
    with plt.style.context("seaborn-poster"):
        fig, ax = plt.subplots(2,1,figsize=(20,17))
        sns.set_palette("muted")
        fig.suptitle("Patients Age Distribution", fontsize=20, fontweight="bold")

        ax1, ax2 = ax.flatten()
        fig.patch.set_facecolor("w")
        ax1.set_facecolor("whitesmoke")
        sns.histplot(d.age,stat="count", binwidth=5,fill=True, linewidth=1,log_scale=(False, True), ax=ax1)

        mean_res = np.round(d.age.mean(),3)
        min_ylim, max_ylim= plt.ylim()
        ax1.axvline(mean_res, color="r", linestyle="dashed", linewidth=2)
        ax1.text(mean_res, max_ylim, "Mean: {:.2f}".format(mean_res), color="red", fontsize=15)

        ax1.set_title(f"Frequency, Mean: {mean_res}")
        ax1.set_xlabel("Patient Age")
        ax1.set_ylabel("Frequency [log scale]")
        #ax1.set_ylim([0,1e4])

        ####
        ax2.set_facecolor("whitesmoke")
        sns.histplot(data=d, stat="count", multiple="stack",binwidth=5,
                 x="age", kde=False,
                 hue="sex_cat",
                 element="bars", fill=True, linewidth=1, log_scale=(False, True),legend=True, ax=ax2)
        ax2.set_title(f"Frequency per Sex")
        ax2.set_xlabel("Patient Age")
        ax2.set_ylabel("Frequency [log scale]")

        plt.show()
        
        
def lineplot_ecg(waveform, rpeaks, heart_rate):
    time_wf = np.arange(waveform.shape[0])*1/100
    
    with plt.style.context("seaborn-poster"):
        fig, ax = plt.subplots(2,1,figsize=(25,16))
        #fig.suptitle("Frequency [%] of Diagnostic Superclasses per Sex", fontsize=20, fontweight="bold")
        ax1, ax2 = ax.flatten()
        fig.patch.set_facecolor("w")
        
        ax1.set_facecolor("whitesmoke")
        ax1.set_title("ECG and Peaks", fontsize=18)
        sns.lineplot(x=time_wf,
                     y=waveform, label="ECG signal", ax=ax1)
        sns.scatterplot(time_wf[rpeaks], waveform[rpeaks], color='r', label="Peaks",ax=ax1)
        ax1.set_ylabel("Voltage ($\mu$V)")
        ax1.set_xlabel("Time [s]")

        ax2.set_facecolor("whitesmoke")
        ax2.set_title("Heart Frequency", fontsize=18)
        sns.lineplot(x=time_wf[rpeaks][1:],
                     y=heart_rate, color="g", marker="o",label="Heart Rate", ax=ax2)
        sns.lineplot(x=time_wf[rpeaks][1:],
                     y=np.mean(heart_rate), color="b",
                      label="Mean Heart Rate", ax=ax2)
        sns.lineplot(x=time_wf[rpeaks][1:],
                     y=np.mean(heart_rate)-np.std(heart_rate), color="b",
                     linestyle='--', label="- Std Heart Rate",alpha=0.6, ax=ax2)
        sns.lineplot(x=time_wf[rpeaks][1:],
                     y=np.mean(heart_rate)+np.std(heart_rate), color="b",
                     linestyle='--',alpha=0.6, ax=ax2)
        ax2.set_xlabel("Number of ECG cycle")
        ax2.set_ylabel("Bpm per second")
        ax2.set_xlim([0, time_wf[rpeaks][1:].max()+1])
        ax2.set_ylim([min(heart_rate)-2, max(heart_rate)+2])
        plt.tight_layout()
        plt.show()
        
def waveform_plot(ecg_id, ECG, df_db, Fs):
    
    ECG_temp = ECG[ecg_id]
    meta_data = df_db.iloc[ecg_id]
    #r_peaks = np.array(meta_data.r_peaks)
    
    leads = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    leads_index = list(range(13))
    
    time = np.arange(ECG_temp.shape[0])*1/Fs
    
    # Setup figure
    fig = plt.figure(figsize=(12, 12), facecolor='w')
    fig.subplots_adjust(wspace=0, hspace=0.05)
    ax1 = plt.subplot2grid((1, 1), (0, 0))

    # ECG
    ax1.set_title(
        'Patient Id: {}\nAge: {}\nSex: {}\nLabel: {}'.format(
            meta_data.patient_id, meta_data.age, meta_data.sex_cat,
            meta_data.diagnostic_binary_superclass
        ),
        fontsize=18,
        loc='left',
        x=0,
    )
    shift = 0
    for channel_id in range(ECG_temp.shape[1]):
        #print(channel_id)
        ax1.plot(time, ECG_temp[:, channel_id] + shift, '-k', lw=2)
        ax1.text(-0.5, 0.25 + shift, leads[channel_id], fontsize=16, ha='left')
        shift += 3
    plt.show() 
    
    
def QRS_detection_plot(wf,wf_bp, wf_diff, wf_ma,  wf_peaks,dist_r, Fs):
    time_wf =  np.linspace(0,len(wf)/Fs,len(wf))
    with plt.style.context("seaborn-poster"):
        fig, ax = plt.subplots(6,1,figsize=(30,55))
        fig.suptitle("Pan-Tompkins Steps for QRS Detection", fontsize=20, fontweight="bold",y=1.0)
        ax1, ax2, ax3, ax4, ax5, ax6 = ax.flatten()
        fig.patch.set_facecolor("w")
        
        ax1.set_facecolor("whitesmoke")
        ax1.set_title("Raw Signal", fontsize=18, fontweight="bold")
        sns.lineplot(x=time_wf,
                     y=wf, ax=ax1)
        ax1.set_ylabel("Voltage ($\mu$V)")
        ax1.set_xlabel("Time [s]")
        ax1.grid()

        ax2.set_facecolor("whitesmoke")
        ax2.set_title("BandPassed Signal", fontsize=18, fontweight="bold")
        sns.lineplot(x=time_wf,
                     y=wf_bp,  ax=ax2)
        ax2.set_ylabel("Voltage ($\mu$V)")
        ax2.set_xlabel("Time [s]")
        ax2.grid()
        
        ax3.set_facecolor("whitesmoke")
        ax3.set_title("Squared Derivate Signal", fontsize=18, fontweight="bold")
        sns.lineplot(x=time_wf,
                     y=wf_diff, ax=ax3)
        ax3.set_ylabel("Voltage ($\mu V^2$)")
        ax3.set_xlabel("Time [s]")
        ax3.grid()
        
        ax4.set_facecolor("whitesmoke")
        ax4.set_title("MovingAveraged Signal", fontsize=18, fontweight="bold")
        sns.lineplot(x=np.linspace(0,len(wf_ma)/Fs,len(wf_ma)),
                     y=wf_ma,  ax=ax4)
        ax4.set_ylabel("Voltage ($\mu$V)")
        ax4.set_xlabel("Time [s]")
        ax4.grid()
        
        ax5.set_facecolor("whitesmoke")
        ax5.set_title("Raw Signal and R-Peaks", fontsize=18, fontweight="bold")
        sns.lineplot(x=time_wf,
                     y=wf, ax=ax5)
        ax5.vlines(x=(wf_peaks)/100,ymin=np.min(wf),
                   ymax=np.max(wf),linestyles='dashed',color='r'
                   ,linewidth=2.0)
        #sns.scatterplot(wf_peak, wf_peaks, color='r', label="Peaks",ax=ax5)
        ax5.set_ylabel("Voltage ($\mu$V)")
        ax5.set_xlabel("Time [s]")
        ax5.grid()
        
        ax6.set_facecolor("whitesmoke")
        ax6.set_title("Raw Signal and QRS", fontsize=18, fontweight="bold")
        sns.lineplot(x=time_wf,
                     y=wf, ax=ax6)
        ax6.vlines(x=(wf_peaks)/100,ymin=np.min(wf),
                   ymax=np.max(wf),linestyles='dashed',color='r'
                   ,linewidth=2.0)
        ##
        ax6.vlines(x=(np.array(dist_r))/100,ymin=np.min(wf),
                   ymax=np.max(wf),linestyles='dashed',color='g'
                   ,linewidth=2.0)
        
        ax6.set_ylabel("Amp")
        ax6.set_xlabel("Time [s]")
        ax6.grid()
        
        
        #ax2.set_xlim([0, time_wf[rpeaks][1:].max()+1])
        #ax2.set_ylim([min(heart_rate)-2, max(heart_rate)+2])
        plt.tight_layout()
        plt.show()