# ZGaming: Zero-Latency 3D Cloud Gaming by Image Prediction

ZGaming is a novel 3D cloud gaming system based on image prediction, in order to eliminate the interactive
latency in traditional cloud gaming systems. This repository is the implementation of ZGaming's image prediction algorithm, specifically divided into foreground prediction and background prediction.

![teaser1](docs/imgs/main_pic.jpg)

## Foreground Prediction
### (0) Getting Started
Clone this repository, enter the `'foreground'` folder and create local environment: `conda env create -f environment.yml`.
### (1) Pre-trained models
Download the pre-trained models from [here](https://drive.google.com/drive/folders/129ftjmfHjoehxGyi6HLQszVRgVN_WBGV?usp=sharing), and place them in the `'checkpoints'` folder.
### (2) Dataset
As a demo, we provide some data in the `'sample'` folder, which you can test directly. For a complete evaluation, you need to obtain the full dataset from [here](https://github.com/ZheC/GTA-IM-Dataset).
### (3) Test
```bash
$ python run.py
```
The predicted images are saved in the `'results'` folder.

Note that in the default configuration, ZGaming will resize the input images to a resolution of `256*256` for prediction. This is because predicting high-resolution images requires a huge amount of GPU memory (for example, a resolution of `1024*1024` requires `41GB`). If your GPU meets these requirements, you can change `'--img_width'` in `'run.py'` to `1024`, and change `'--pretrained_model'` in `'run.py'` to `'./checkpoints/in5_out10_1024_60k.ckpt'`.

## Background Prediction
### (0) Getting Started
Clone this repository, enter the `'background'` folder.

Configure OpenCV for C++ (We tested with OpenCV 4.5.5) and compile the project.

### (1) Dataset
As a demo, we provide sample data, which you can test directly. Please download the zip file from [here](https://drive.google.com/file/d/19gWhURb2LU7ysBNeMSUeIe1Il6VLXjO7/view?usp=drive_link), unzip it and place the `'2020-06-03-20-28-01'` and `'2020-06-03-22-25-09'` folders in the `'data'` folder.

For a complete evaluation, you need to obtain the full dataset from [here](https://github.com/ZheC/GTA-IM-Dataset).
### (2) Test
Run the compiled project. This demo will first cache a short game video, then perform DIBR predictions with and without cache separately. 

The predicted images will be saved in the `'results'` folder, and the accuracy results in PSNR will be saved in `'results\log.txt'`.

## Acknowledgement
Our prediction algorithms are built based on these three repositories:

[predrnn-pytorch](https://github.com/thuml/predrnn-pytorch) ![GitHub stars](https://img.shields.io/github/stars/thuml/predrnn-pytorch.svg?style=flat&label=Star)

[MASA-SR](https://github.com/dvlab-research/MASA-SR) ![GitHub stars](https://img.shields.io/github/stars/dvlab-research/MASA-SR.svg?style=flat&label=Star)

[GTA-IM-Dataset](https://github.com/ZheC/GTA-IM-Dataset) ![GitHub stars](https://img.shields.io/github/stars/ZheC/GTA-IM-Dataset.svg?style=flat&label=Star)