# Image segmentation using Deep Learning

## Prerequisites
From the major dependencies project requires Keras and Tensorflow. Installation quick guides:

- [TensorFlow installation instructions](https://www.tensorflow.org/install/)
- [Keras installation instructions.](https://keras.io/#installation)

## Installation
Create virtual environment and install requires packages.
```
pip install -r requirements.txt
```
It might be needed to install requirements as user (add `--user` flag to pip)

## Usage
Run `main.py` and you will see the predicted results of test image in data/membrane/test. Alternatively you can follow notebook [trainUnet.ipnb](./trainUnet.ipnb)


## Goal
The goal is to train the network to distinguish different parts of input image, in other words to perform segmentation. The approach used here is to train deep neural network to conduct segmentation task. The network architecture is U-Net as proposed in: [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).


## Data

The experiments were conducted on dataset from [isbi challenge](http://brainiac2.mit.edu/isbi_challenge/). He is how the input data looks like:

<img src="./data/membrane/train/image/0.png" width="25%">
*Figure 1. Input image that needs segmentation.*

<img src="./data/membrane/train/label/0.png" width="25%">
*Figure 2. Labels for training - in black pixels considered as boundaries.*

The input data for the training is the 30 images which is much too less to train the neural network. To proceed, available data was transformed in different ways to produce new samples that can be used for training. Note, that is labeling data need to be transformed in the same way as input image (features).


**Data augmentation**
The data for training contains 30 images 512x512 pixels each. Fortunately, this is quite common problem in deep learning for computer vision and Keras module `ImageDataGenerator` in `keras.preprocessing.image` has already functions to make different kinds of transformations.

**Data augmentation - resulting samples**


## Model

![Neural network architecture](img/u-net-architecture-hi-res.png)

This deep neural network is implemented with Keras functional API, which makes it extremely easy to experiment with different interesting architectures.

Output from the network is a 512x512 which represents mask that should be learned. Sigmoid activation function makes sure that mask pixels are in \[0, 1\] range.

## Training

* The model is trained for 5 epochs. After 5 epochs, calculated accuracy is about 0.97.

* Loss function for the training is  a **binary crossentropy**.

## Results

Use the trained model to do segmentation on test images, the result is statisfactory.



## Credits:
* *Olaf Ronneberger, Philipp Fischer, Thomas Brox* for Unet Convolutional Network
* implementation is heavily inspired by: [zhixuhao](https://github.com/zhixuhao/unet) 