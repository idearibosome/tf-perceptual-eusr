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
- Extract the zipped file to a desired location. Here we assume that the extracted location is ```/tmp/DIV2K/train/HR```.
- Open MATLAB, specify the working directory as ```train/misc/```, and run the following MATLAB commands:
```
downscale_generator_matlab('/tmp/DIV2K/train/HR', '/tmp/DIV2K/train/LR/x2', 2);
downscale_generator_matlab('/tmp/DIV2K/train/HR', '/tmp/DIV2K/train/LR/x4', 4);
downscale_generator_matlab('/tmp/DIV2K/train/HR', '/tmp/DIV2K/train/LR/x8', 8);
```

Now, the directory structure should be looked like this:
```
/tmp/DIV2K/train/
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
You can also perform the same procedure for the validation images.

We also provide alternative downscaling helper scripts written in Python: ```downscale_generator_cv.py``` (downscale via the OpenCV function) and ```downscale_generator_tf.py``` (downscale via the TensorFlow function).
However, please note that the downscaled results may not exactly the same as the images obtained from MATLAB.


### Pre-training the EUSR model
The EUSR model is implemented and refactored from its [official TensorFlow-based implementation](https://github.com/junhyukk/EUSR-Tensorflow).

Here is an example command to train the EUSR model:
```
python train.py
  --data_input_path=/tmp/DIV2K/train/HR
  --data_truth_path=/tmp/DIV2K/train/LR
  --train_path=/tmp/tf-perceptual-eusr/eusr
  --model=eusr
  --scales=2,4,8
```
You can also change other parameters, e.g., the maximum number of training steps and learning rate.
Please run `python train.py --model=eusr --helpfull` for more information.

During the training, you can view the current training status via TensorBoard, e.g.,
```
tensorboard --logdir=/tmp/tf-perceptual-eusr/eusr
```

You can also validate the trained model by ```validate.py```.
For example, if you want to evaluate the model saved at step 50000, run
```
python validate.py
  --data_input_path=/tmp/DIV2K/validate/HR
  --data_truth_path=/tmp/DIV2K/validate/LR
  --model=eusr
  --scales=2,4,8
  --restore_path=/tmp/tf-perceptual-eusr/eusr/model.ckpt-50000
  --save_path=/tmp/tf-perceptual-eusr/eusr/results
```
It will print out the PSNR and RMSE values of the upscaled images with saving them on the path that you specified in ```--save_path```.
Please run `python validate.py --model=eusr --helpfull` for more information.

※ Note that the calculated PSNR and RMSE values may differ from the the values in our paper, due to the different calculation methods.
The code in this repository calculates PSNR and RMSE values from R, G, and B channels, while the measures reported in the paper were measured from Y channel of the YCbCr color space.


## TODO
- Freezing the trained model
- Training qualitative score predictors
- Training the 4PP-EUSR model
