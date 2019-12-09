# uDoppler-Classification
Classification of Human Movement using mmWave FMCW Radar Micro-Doppler Signature

## Problem Statement
The sensor suites needed for autonomous machines to function include varying assortments of sensing modalities. These modalities include LiDAR, RADAR, Cameras, Ultra-sonic, etc. Traditionally, LiDAR and Cameras have provided valuable imaging capabilities. However, there are inherent issues with these sensors that make them unideal of some use-cases, and useless in other use-cases. Namely,

1. Camera based classification doesn't work well when FOV is dimly lit
2. Solid State LiDAR systems are still in development. Mechanical LiDAR systems are large and prone to mechanical wear-and-tear

If a different sensing modality could be used to classify targets on the road with high accuracy and are impervious to the issues cameras and LiDAR face, then autonomous systems can greatly improve their safety.

Should a solution like this exist, it would need to do a few things.

1. Accurately classify common obstacles.
2. Separate objects that are close in the sensor's FOV
3. Track objects as their move in the sensor's FOV

## Our Solution
Millimeter wave (mmwave) radar hardware has been following a few trends.

1. Form factor is decreasing
2. Cost is decreasing
3. Imaging capabilities are increasing

Using the latest Texas Instrument mmwave radar system, can we classify targets based on their microdoppler signatures alone?

For our project, we will be using the Texas Instrument 1843 mmwave multiple-input-multiple-output (MIMO) mmwave radar system. As a MIMO system, we will achieve source separation using the angle of arrival (AOA), and range as differentiating factors. Tracking of targets in the radar FOV will be done with an Extended Kalman Filter (EKF). Classification will then be done on the separated target microdoppler signatures. For this experiment, we will attempt to classify targets as pedestrians or bicycles. 

## Design

### Source Separation

### Tracking

### Classification
For this experiment, we attempt to classify microdoppler signature spectrograms as either orginating from a pedestrian or a bicycle. With each 2 second long scan, we generate on spectrogram. This spectrogram is treated like an image.

Classification is done using the following two techniques:
1. Gaussian Mixture Model (GMM)
2. Convolutional Neural Net (CNN)

Hidden Markov Models were considered...
