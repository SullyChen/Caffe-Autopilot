# Caffe-Autopilot

**This software is for demonstration purposes only. Please _DO_ _NOT_ use this to pilot a car!**

A car autopilot system developed using C++, [BVLC Caffe](https://github.com/BVLC/caffe), [OpenCV](http://opencv.org/), and [SFML](http://www.sfml-dev.org/)

**The model in this repository has _only_ been trained on roads with a visible, solid yellow line divider. More models are being trained for different road types and driving conditions. These models will be uploaded when they are ready.**

[Video here](https://www.youtube.com/watch?v=fSbWnQ_wzvM)

[Datasets and models](https://drive.google.com/open?id=0B-KJCaaF7ellNFVFSUpVWGlTUWM)

## How it works

Deep convolutional neural nets are powerful.

First, a dataset is collected using a webcam and an Arduino with a CANBUS shield to read the steering wheel angles from the OBD-II port of the car.

The images are preprocessed with Canny edge filtering and a certain threshold operation as they are collected.
The purpose of the preprocessing is to make less work for the convolutional neural network by extracting the important features with image processing algorithms ahead of time.

Then, [BVLC Caffe](https://github.com/BVLC/caffe) is used to train a deep convolutional neural network (specifically AlexNet).

To predict steering angles based on input images, a weighted sum of the BVLC Caffe classifier outputs are used to calculate the final angle. Some smoothing algorithms are also used to smooth the motion of the predicted steering wheel angle.
