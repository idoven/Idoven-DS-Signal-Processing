---
title:  Idoven Interview Challenge
author: Killian Smith
date:   December 9-16, 2022
---

# About

## Install

To configure the project (data retrieval, docker image, python environment), run
the following set of commands from the directory `smith_killian_subbmission/`.

- `make get-data` 
- `sudo make docker-volume`
- `sudo make build`

At this point you should be able to start the docker image with `sudo make
start`. To verify this, run `sudo make status`. You should see a running docker
instance. The notebook should now be available in your browser from:
`http://localhost:8888/notebooks/project/notebooks/idoven-challenge.ipynb`.
When finished, run `sudo make stop` to close the docker image.

# Log

## Day 1 (Friday)

Received the challenge in the morning, but had to wait to start until the
evening as I had to go into the office for my current job.

This evening's task will be to get `docker`, `jupyter`, and `conda` playing nice
with each other. This will be the first time I have really used `docker`or
`jupyter` seriously (normally I use `nixpkgs+flakes` for reproducibility and
`org-mode` for notebooks), so I will start from an existing
configuration. After a quick search, I found an interesting template from
[@borundev][borundev-git-repo] that allowed for multiple `conda`
environments. There were a lot of extra fancy things in the repo, but I am just
starting out so I will strip out everything except for the `conda`/`jupyter`
parts.

The next thing to do is set up a data volume for the docker machine. I do not
really want to be copying the dataset back and forth, so I will instead make a
docker volume and share the directory on the host machine. 

Since I am not on the best internet connection, I will download the compressed
ptb-xl-v1.0.2 dataset instead of using the provided retrieval script.  There is
a newer version of the data available (v1.0.3), but for consistency I will use
the v1.0.2 version as given in the instructions ([PTB-XL][PTB-XL-v1.0.2.zip]).

At this point it looks like everything is working, and I have a `jupyter`
notebook at [notebook][idoven-notebook]. I will stop here for tonight.

## Day 2 (Saturday)

Okay, first full day to work. Lets get started!

Today will be exploring the dataset, filtering out anything that looks
erroneous, and (if we need to) down-sample to something that is manageable on a
personal laptop. 

The **PTB-XL** data includes a small python script for loading the database file
and the ECGs. This is pretty nice, and I will use it as a starting point.

Most of the data looks pretty clean. Any rows that did not have a label,
reported noise, had a pacemaker, or were not validated by a human, were filtered
out. This should leave us with a pretty clean dataset to run our classifier on.

Speaking of classifier, we are probably going to try to create and train some
Convolutional Neural Networks (CNN), and we need to get our labels into
something more usable than a list of strings. I added an extra column for a
*one-hot-encoding* of the labels. At some point it may be a good idea to add a
second column for a *"normal"/"abnormal"* *one-hot-encoding* to use for an
easier starter CNN.

The dataset also needs to be split into train/validate/test partitions, and we
need to double check the distributions of the labels in each of the sets to make
sure no biases are introduced.

This seems like a good spot to stop for the day, and think about what are the
next steps.

## Day 3 (Sunday)

Today is going to be all about ECGs!

In the morning I spent time reading up a bit on what ECGs are, what are the
important features (P, QRS, T), data format and storage, and what other
researchers were using to do feature extraction.

To be honest, I got really distracted by the wavelet transform (and wavelet
convolutions in general) at this point. This was the first time I was hearing
about these wavelet techniques, and they are so cool!

Once I got back on track, I started with building a function for plotting all 12
leads in a readable way. I grabbed a normal ECG and 2 ECGs with different
pathologies to try and look to see if I could see anything. I could not really,
so it is probably good I am not a cardiologist. 

Next step was to do the same thing, but with the FFT of each of the ECGs. I
thought maybe there might be some differences in the frequency domain that could
be used. Did a bit of reading and learned that the `scipy` fft methods were
preferred to the semi-deprecated `numpy` fft. In the end, this turned out to be
even less helpful than the ECGs.

At the very end of the day I started with the scalograms. Looks cool, but is
much more involved.

## Day 4 (Monday)

Have to go back to work in the office again, so only was able to work in the
evening. But this was a good day; I finally got the settings right on the
scalograms and even I can see the differences in the pathologies! It is so cool! 

Played around a bit with different settings of the wavelet and scale parameters,
but the scalograms look like they are very actionable. The next step would be
training a CNN for a single lead, and then scaling it up to a graph neural
network with 12 CNNs (one for each lead) feeding into a regular dense network?
It could be cool, but maybe a pain in the ass to train? Something to try :)

## Day 5 (Tuesday)

Working in the office again today, and only have a bit of time in the
evening. Rest of the week is really busy (giving a seminar, in the middle of
moving, and have a flight to pack for). This will be my last day working on the
challenge for now.

Today I just went through the docker build process from scratch, and run the
jupyter notebook from a clean state to double check that everything is ready to
submit.

Last thing to do is make a git branch, copy the project contents, and make a
pull request.

There are still so many things that I wanted to do for this challenge, but did
not have time for. The main points that I still want to be working on:

  - Understand the scale parameter in the scalograms
  - Test each of the continuous wavelets on each of the pathologies
  - Look to see if some leads are more useful than others
  - Build a CNN for a single lead
  - Try to train a GNN (DAG of 12 CNN feeding into a dense NN)

## Day 6 (Wednesday)

Too busy, and could not work on project.

## Day 7 (Thursday)

Double check documentation and submit the pull request.

# Future Research Questions 

This was a very fun and interesting interview challenge, but not nearly enough
time to explore all of the possibilities!

## Pathologies vs ECG Lead

It would be very interesting to see if there is a difference in what pathologies
each lead can detect. If there is a significant difference in which pathologies
could be detected on a lead, it could be worthwhile to partition the CNN and
train subsets of leads separately (much more efficient).

## Wavelet and Scale Choice

There are *many* wavelets to choose from when building our scalograms. It would
be very interesting to see what the effects of wavelet choice are on the
classifier, and if certain pathologies can be classified with a lower error rate
using a particular wavelet.

Also, at the moment I have no idea how to be choosing the scale factors of the
wavelets. In theory there should be a way of mapping between frequencies and
scales, but this is something that I need to think about.

## "2-Pass" System

Most of the labels in our dataset are from *"normal"* ECG readings. It may be
useful to make an initial classifier to distinguish between normal vs abnormal,
and then feed abnormal results to a CNN trained *only* on pathologies. This
should lead to a more sensitive classifier.

## Pathologies vs Data Representation

In theory, using a CNN for scalograms should be the most powerful (uses data
from both the time and frequency domains), but at the same time it is much more
resource intensive to train. If some pathologies could be detected with high
accuracy from a CNN trained on either the ECG or the FFT, it would be better to
use these simpler models.


# Software Stack

I work on many different projects using different languages and software stacks,
so I am very bad about getting function names, data structure types, and
argument ordering. For this reason I like to keep a list of links to all APIs
and external libraries that I am using when developing. 

  - [python](https://docs.python.org/3/)
  - [numpy](https://numpy.org/doc/stable/)
  - [scipy](https://docs.scipy.org/doc/scipy/)
  - [wfdb](https://wfdb.readthedocs.io/en/latest/)
  - [pywt](https://pywavelets.readthedocs.io/en/latest/index.html)
  - [pandas](https://pandas.pydata.org/docs/)
  - [matplotlib](https://matplotlib.org/stable/index.html)
  - [seaborn](https://seaborn.pydata.org/api.html)
  - [scikit-learn](https://scikit-learn.org/stable/modules/classes.html)
  - [tensorflow](https://www.tensorflow.org/api_docs/python/tf)
  - [keras](https://keras.io/api/)

# References

This is a very new domain for me, and I have a lot of literature to catch up
on. The following is a short list of papers that I found helpful during the
challenge. 

In addition, I found the youtube channel of *Mike X Cohen* where he has a series
of lectures about doing analyses on EEG datasets. It has some very nice visuals
and is helpful to understand the basics of signal analysis in the biomedical
field. I think this should transfer to ECG data quite well.

Here is the list of publications that were skimmed over or need to be read in
the future:

## A Patient-Adapting Heartbeat Classifier Using ECG Morphology and Heartbeat Interval Features
Authors: Chazal, et. al.
DOI: 10.1109/TBME.2006.883802

## Symmetrical Compression Distance for Arrhythmia Discrimination in Cloud-Based Big-Data Services
Authors: Lillo-Castellano, et.al.
DOI: 10.1109/JBHI.2015.2412175

## Big-data analytics for Arrhythmia Classification using data compression and kernel methods
Authors: Lillo-Castellano, et.al.
DOI: 10.1109/CIC.2015.7410997

## Classification of cardiac arrhythmia using a convolutional neural network and bi-directional long short-term memory
Authors: Hassan, et.al.
DOI: 10.1177/20552076221102766

## ENCASE: An ENsemble ClASsifiEr for ECG classification using expert features and deep neural networks
Authors: Hong, et. al.
DOI: 10.22489/CinC.2017.178-245

## Evolving a Bayesian Classifier for ECG-based Age Classification in Medical Applications
Authors: Wiggins, et. al.
DOI: 10.1016/j.asoc.2007.03.009

## Reduced-Lead ECG Classifier Model Trained with DivideMix and Model Ensemble
Authors: Seki, et. al.
DOI: 10.23919/CinC53138.2021.9662858

## Identifying normal, AF and other abnormal ECG rhythms using a cascaded binary classifier
Authors: Datta, et. al.
DOI: 10.22489/CinC.2017.173-154

## Design of a Low-Power On-Body ECG Classifier for Remote Cardiovascular Monitoring Systems
Authors: Chen, et. al.
DOI: 10.1109/JETCAS.2013.2242772

## A Fast Machine Learning Model for ECG-Based Heartbeat Classification and Arrhythmia Detection
Authors: Alfaras, et. al.
DOI: 10.3389/fphy.2019.00103

## ECG beat classifier designed by combined neural network model
Authors: Güler and Übeyli
DOI: 10.1016/j.patcog.2004.06.009


# Xref Links

[PTB-XL-v1.0.2.zip]: https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.2.zip
[borundev-git-repo]: https://github.com/borundev/machine_learning_docker
[idoven-notebook]: http://localhost:8888/notebooks/project/notebooks/idoven-challenge.ipynb
