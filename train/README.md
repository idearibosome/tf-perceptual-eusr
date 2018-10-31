# Training 4PP-EUSR
※ Please note that the implementation of training code for public is currently in progress.


## Tutorial
This tutorial demonstrates how to train a 4PP-EUSR model with the codes of this repository.
Briefly, there are three training phases to get a 4PP-EUSR model as follows:
- Pre-training the EUSR model
- Training aesthetic and subjective qualitative score predictors
- Training the 4PP-EUSR model based on the pre-trained EUSR and score predictors


### Preparing training images
Here, the DIV2K dataset will be used for training.
You can download the DIV2K dataset from its [official website](https://data.vision.ee.ethz.ch/cvl/DIV2K/).

The EUSR model is trained with images downscaled with MATLAB's ```imresize``` function by factors of 2, 4, and 8.
However, the origianl DIV2K dataset does not provide the x8 downscaled images.
Therefore, we provide a helper script to generate downscaled images.
Followings are the brief instruction.

- Download the high resolution (HR) images from the official DIV2K website.
- Extract the zipped file to a desired location. Here we assume that the extracted location is ```/tmp/DIV2K/HR```.
- Open MATLAB, specify the working directory as ```train/misc/```, and run the following MATLAB commands:
```
downscale_generator_matlab('/tmp/DIV2K/HR', '/tmp/DIV2K/LR/x2', 2);
downscale_generator_matlab('/tmp/DIV2K/HR', '/tmp/DIV2K/LR/x4', 4);
downscale_generator_matlab('/tmp/DIV2K/HR', '/tmp/DIV2K/LR/x8', 8);
```

Now, the directory structure should be looked like this:
```
/tmp/DIV2K/
|- HR/
   |- 0001.png
   |- 0002.png
   |- ...
|- LR/
   |- x2/
      |- 0001.png
      |- 0002.png
      |- ...
   |- x4/
      |- ...
   |- x8/
      |- ...
```

We also provide alternative downscaling helper scripts written in Python: ```downscale_generator_cv.py``` (downscale via the OpenCV function) and ```downscale_generator_tf.py``` (downscale via the TensorFlow function).
However, please note that the downscaled results may not exactly the same as the images obtained via MATLAB.


### Pre-training the EUSR model
The EUSR model is implemented and refactored from its [official TensorFlow-based implementation](https://github.com/junhyukk/EUSR-Tensorflow).

You can try out our EUSR training code by running the following command:
```
python train.py --data_input_path=[data_input_path] -data_truth_path=[data_truth_path] --train_path=/tmp/tf-perceptual-eusr/eusr --model=eusr --scales=2,4,8
```
Please run `python train.py --model=eusr --helpfull` for more information.
More detailed instructions will be available when our training code is further implemented.


## TODO
- Freezing the trained model
- Training qualitative score predictors
- Training the 4PP-EUSR model
