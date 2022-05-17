# TSMF-Net

This is a implementation of TSMF-Net under the Pytorch framework



## Requirements

| Environment & Package | Version  |
| :-------------------: | :------: |
|        python         |  3.6.10  |
|         cuda          |   10.1   |
|         torch         |  1.3.1   |
|      torchvision      |  0.4.2   |
|        opencv         | 4.4.0.46 |
|         gdal          |  3.0.2   |
|        libtiff        |  0.4.2   |
|         numpy         |  1.19.2  |
|        pillow         |  8.0.1   |
|         scipy         |  1.5.4   |
|      hdf5storage      |  0.1.18  |

Before you start running our code, make sure that you have installed the various libraries above and in `requirements.txt` as required. Our code runs on Linux system. To avoid unnecessary problems, please test our code on Linux systems.



## Preprocess

1. Prepare the PAN and MS images and the ground truth file
2. Ensure the format of images is `.tiff` or `.tif`, and the format of ground truth is `.npy`
3. Calculate the linear factors of the PAN and MS images by `factors.m`
4. Fuse the PAN and MS images by `image_fusion.py`
5. Storage the `.npy` fusion results with the ground truth in the same folder



## Train&Test

1. Prepare your PAN and MS images and the ground truth file
2. Preprocess the data
3. Modify the parameters of the codes to fit your need
4. Run the main scripts to train or test your model
5. Save the weights to  `.pkl` file and have a test



## Visualize

1. Prepare your PAN and MS images and the ground truth file
2. Preprocess the data
3. Ensure your model in `.pkl` format
4. Modify the parameters of the codes to fit your need
5. Visualize the results by `ms_visual.py` and `pan_visual.py`



