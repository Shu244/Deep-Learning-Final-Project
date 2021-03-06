
# MaskTrack RCNN

## Introduction

In this work, a new task video instance segmentation is presented. Video instance segmentation extends the image instance segmentation task from the image domain to the video domain. The new problem aims at **simultaneous detection, segmentation and tracking** of object instances in videos.
We also proposed an algorithm to jointly detect, segment, and track object instances in a video, named MaskTrackRCNN.  A tracking head is added to the original MaskRCNN model to match objects across frames.
## Installation
This repo is built based on [mmdetection](https://github.com/open-mmlab/mmdetection) commit hash `f3a939f`. Please refer to [INSTALL.md](INSTALL.md) to install the library.
You also need to install a customized [COCO API](https://github.com/youtubevos/cocoapi) for dataset.
You can use following commands to create conda env with all dependencies.
```
conda create -n MaskTrackRCNN -y
conda activate MaskTrackRCNN
conda install -c pytorch pytorch=0.4.1 torchvision cuda92 -y
conda install -c conda-forge cudatoolkit-dev=9.2 opencv -y
conda install cython -y
pip install git+https://github.com/youtubevos/cocoapi.git#"egg=pycocotools&subdirectory=PythonAPI"
bash compile.sh
pip install .
```
You may also need to follow [#1](/../../issues/1) to load MSCOCO pretrained models.
## Model training and evaluation
Our model is based on MaskRCNN-resnet101-FPN. The model is trained end-to-end on our dataset based on a MSCOCO pretrained checkpoint.
### Training
1. Symlink the train/validation dataset to `$MMDETECTION/data` folder. Put COCO-style annotations under `$MMDETECTION/data/annotations`.
```
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── train
│   ├── val
│   ├── annotations
│   │   ├── instances_train_sub.json
│   │   ├── instances_val_sub.json
```

2. run main.ipynb to tran the model. 

### Evaluation

The results are stored in the vis.zip. You could download and see the tracking results.

## Contact
If you have any questions regarding the repo, please contact Luojie Huang <lhuang48@jh.edu>.

