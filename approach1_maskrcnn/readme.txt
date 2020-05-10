Group 9: Deep Learning for Non-Invasive Blood Cytometry
Group members: Yuan Zhou, Shuhao Lai, Luojie Huang

* This package 'approach1_maskrcnn' can be implemented on Google Colab. Please make sure the whole package is uploaded to Google Drive before running. Also, the test data 'GFC_505nm_40X1p15NA_PCO_500usexp_160fps_3@0001.tif' should be put in 'approach1_maskrcnn'.

This package includes:
	1. Model: 'Mask RCNN' folder
	2. Preprocession: pre_stab.py, stabilization.ipynb, 's2' folder , 'stab' folder
	3. Cell counting: CellCount.ipynb

Descriptions:
	1. 'Mask RCNN' includes all the codes (config.py, model.py, utils.py, visualize.py) for our used model in the first approach (not including our trained model due to its big size).
	2. pre_stab.py and stabilization.ipynb contain the codes for video stabilization.
	3. 's2' folder is the path for all the frame images from the test data without stabilization while 'stab' folder is for all the stabilized images.
	4. CellCount.ipynb is used for training and testing our Mask RCNN as well as the cell counting step.