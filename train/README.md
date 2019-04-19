# Training 4PP-EUSR
※ Please note that the implementation of training code for public is currently in progress.

This tutorial demonstrates how to train a 4PP-EUSR model with the codes of this repository.
Briefly, there are three training phases to get a 4PP-EUSR model as follows:
- Pre-training the EUSR model
- Training aesthetic and subjective qualitative score predictors
- Training the 4PP-EUSR model based on the pre-trained EUSR and with employing the score predictors


## Preparing training images
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


## Pre-training the EUSR model
The EUSR model is implemented and refactored from its [official TensorFlow-based implementation](https://github.com/junhyukk/EUSR-Tensorflow).

Here is an example command to train the EUSR model:
```
python train.py
  --data_input_path=/tmp/DIV2K/train/LR
  --data_truth_path=/tmp/DIV2K/train/HR
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
  --data_input_path=/tmp/DIV2K/validate/LR
  --data_truth_path=/tmp/DIV2K/validate/HR
  --model=eusr
  --scales=2,4,8
  --restore_path=/tmp/tf-perceptual-eusr/eusr/model.ckpt-50000
  --save_path=/tmp/tf-perceptual-eusr/eusr/results
```
It will print out the PSNR and RMSE values of the upscaled images with saving them on the path that you specified in ```--save_path```.
Please run `python validate.py --model=eusr --helpfull` for more information.

※ Note that the calculated PSNR and RMSE values may differ from the the values in our paper, due to the different calculation methods.
The code in this repository calculates PSNR and RMSE values from R, G, and B channels, while the measures reported in the paper were obtained from Y channel of the YCbCr color space.


## Training qualitative score predictors
Our model requires two qualitative score predictors, which are trained on the [AVA](https://ieeexplore.ieee.org/document/6247954) and [TID2013](http://www.ponomarenko.info/tid2013.htm) datasets.
You can train the score predictors manually with the provided code, or skip it and use our pre-trained predictors.
- Aesthetic score predictor (score_predictor_ava.pb) (Coming soon)
- Subjective score predictor (score_predictor_tid2013.pb) (Coming soon)

The training code in ```score_predictors/``` is based on Keras in TensorFlow.
It is basically a refactored and modified version of [https://github.com/titu1994/neural-image-assessment](https://github.com/titu1994/neural-image-assessment).
To train the models, you need TensorFlow 1.11+, since the MobileNetV2 model is included in that version.

### Aesthetic score predictor
The aesthetic score predictor can be trained on the AVA dataset.

- Download and extract the AVA dataset. The dataset should contain ```AVA.txt```, which has both the image ids and scores.
- Prepare the original images of the AVA dataset.
- Run the following code to train the last layer of MobileNetV2, which produces ```ava_lastonly.h5```:
```
python train.py
  --dataloader=ava
  --ava_dataset_path=<path of AVA.txt>
  --ava_image_path=<path of the images>
  --mobilenetv2_train_last_only
  --batch_size=128
  --epochs=5
  --learning_rate=0.001
  --train_path=train
  --weight_filename=ava_lastonly.h5
```
- Notes:
  - The data loader may search all jpg files in ```--ava_image_path``` recursively.
  - The data loader assumes that all the image files are not corrupted. If some image files are corrupted, manually remove the corrupted files or specify ```--ava_validate_images```. However, training with the ```--ava_validate_images``` option may take a while because it tries to read all the image files to check validity.
- Run the following code to fine-tune all the layers, which produces ```ava.h5```:
```
python train.py
  --dataloader=ava
  --ava_dataset_path=<path of AVA.txt>
  --ava_image_path=<path of the images>
  --batch_size=32
  --epochs=5
  --learning_rate=0.00001
  --train_path=train
  --weight_filename=ava.h5
  --restore_path=train/ava_lastonly.h5
```
- Freeze the model to be used in training the 4PP-EUSR model:
```
python freeze.py
  --restore_path=train/ava.h5
  --output_path=train/ava.pb
```

### Subjective score predictor
The subjecitve score predictor can be trained on the TID2013 dataset.

- Download and extract the distorted images.
- Run the following code to train the last layer of MobileNetV2, which produces ```tid2013_lastonly.h5```:
```
python train.py
  --dataloader=tid2013
  --tid2013_image_path=<path of the distorted images>
  --mobilenetv2_train_last_only
  --batch_size=128
  --epochs=100
  --learning_rate=0.001
  --train_path=train
  --weight_filename=tid2013_lastonly.h5
```
- Run the following code to fine-tune all the layers, which produces ```tid2013.h5```:
```
python train.py
  --dataloader=tid2013
  --tid2013_image_path=<path of the distorted images>
  --batch_size=32
  --epochs=100
  --learning_rate=0.00001
  --train_path=train
  --weight_filename=tid2013.h5
  --restore_path=train/tid2013_lastonly.h5
```
- Freeze the model to be used in training the 4PP-EUSR model:
```
python freeze.py
  --restore_path=train/tid2013.h5
  --output_path=train/tid2013.pb
```


## TODO
- Freezing the trained model
- Training the 4PP-EUSR model
