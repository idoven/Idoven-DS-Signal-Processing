# Solution to the data science task

## Usage

This solution can be run in a Docker container or in Colab.

### Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fdavidcl/idoven-task/blob/fdavidcl-project/workspace/project.ipynb)

Please notice that you will need to upload the `requirements.txt` and `best_checkpoint.h5` files to the environment created by Colab.

### Docker

A `docker-compose.yml` file has been provided for convenience. Use the following command to start the notebook server:

```sh
docker-compose up -d
```

The notebook server should appear at `https://localhost:8888`. By default, the container will attempt to use GPU resources from the host. If Docker Compose is not available, a `run_docker.sh` script which launches the same image has been provided as well.

## Reference documentation

### About the data domain

- [PTB-XL, a large publicly available electrocardiography dataset](https://physionet.org/content/ptb-xl/1.0.2/). The source of the dataset.
- [QRS complex - Wikipedia](https://en.wikipedia.org/wiki/QRS_complex). Information on the parts that compose the QRS complexes in an ECG.
- [ECG rate interpretation](https://litfl.com/ecg-rate-interpretation/). Explains how to compute heart rate according to the space between QRS complexes.
- [Table 5 SCP-ECG acronyms](https://www.nature.com/articles/s41597-020-0495-6/tables/6). Meanings of the acronyms for each superclass.
- [The Cabrera format of the 12-lead ECG](https://ecgwaves.com/topic/12-lead-ecg-cabrera-format-inverting-lead-avr/). Explanation of the Cabrera format which is recommended by some cardiology associations to display ECG data.

### About ECG processing and multi-label classification

- [Deep learning for ECG segmentation](https://arxiv.org/pdf/2001.04689.pdf). Using U-Net, a popular CNN for segmentation, for detecting QRS complexes.
- [Pan-Tompkins algorithm](https://en.wikipedia.org/wiki/Pan%E2%80%93Tompkins_algorithm). A very popular QRS detection algorithm.
- [What Is an Arrhythmia?](https://www.nhlbi.nih.gov/health/arrhythmias). US NIH resource on arrhythmias.
- [Addressing imbalance in multilabel classification: measures and random resampling algorithms](https://www.sciencedirect.com/science/article/abs/pii/S0925231215004269). A deep dive into the potential problems of imbalance in a multilabel problem. Also explains which evaluation metrics are more affected by imbalance and which are less so.

### Libraries and tools for implementation

- [wfdb documentation](https://wfdb.readthedocs.io/en/latest/processing.html#module-3). Reading and processing ECG data.
- [Matplotlib API Reference](https://matplotlib.org/stable/api/). Plots in general.
- [pyCirclize](https://moshi4.github.io/pyCirclize/plot_api_example/#1-5-link). Circular plots with tracks.
- [mldr circos plot](https://github.com/fcharte/mldr/blob/master/R/graphics.R#L91), [mldr Shiny app](https://fdavidcl.shinyapps.io/mldr/). Circular plots applied to multi-label datasets.
- [ECG - NeuroKit2](https://neuropsychology.github.io/NeuroKit/functions/ecg.html#analysis). Library for ECG signal processing, QRS peak detection and heart rate computation.
- [Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline](https://arxiv.org/pdf/1611.06455.pdf). Essential deep neural networks for one-dimensional data, including a residual network ([author's implementation](https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline/blob/master/ResNet.py)).
- [Scikit-Learn metrics](https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics). Evaluation metrics for classifiers (multi-label classifiers are evaluated by setting the `average` parameter in some of these).
- [Lime for Time](https://github.com/emanuel-metzenthin/Lime-For-Time/blob/master/lime_timeseries.py). Adaptation of the LIME explanation method for time series.