# Training 4PP-EUSR
Please note that the implementation of training code for public is currently in progress.


## Overview
Briefly, there are three training phases to get a 4PP-EUSR model as follows:
- Pre-training the EUSR model
- Training aesthetic and subjective qualitative score predictors
- Training the 4PP-EUSR model based on the pre-trained EUSR and score predictors


## Completed implementations

### Pre-training the EUSR model
The EUSR model is implemented and refactored from its [official TensorFlow-based implementation](https://github.com/junhyukk/EUSR-Tensorflow).

You can try out our EUSR training code by running the following code:
```
python train.py --data_input_path=[data_input_path] -data_truth_path=[data_truth_path] --train_path=/tmp/tf-perceptual-eusr/eusr --model=eusr --scales=2,4,8
```
Please run `python train.py --helpfull` for more information.
More detailed instructions will be available when our training code is further implemented.


## TODO
- Resuming the training
- Validating the trained model
- Freezing the trained model
- Training qualitative score predictors
- Training the 4PP-EUSR model
