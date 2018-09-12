# 4PP-EUSR
Four-pass perceptual super-resolution with enhanced upscaling


## Introduction
This repository contains a TensorFlow-based implementation of 4PP-EUSR, which considers both the quantitative and perceptual quality of the upscaled images.
Our method won the 2nd place for Region 2 in the [PIRM Challenge on Perceptual Super Resolution at ECCV 2018](https://www.pirm2018.org/PIRM-SR.html).

![BSD100 - 37073](figures/bsd100_37073.png)
※ The perceptual index is calculated by "0.5 * ((10 - [Ma](https://sites.google.com/site/chaoma99/sr-metric)) + [NIQE](https://doi.org/10.1109/LSP.2012.2227726))", which is used in the [PIRM Challenge](https://www.pirm2018.org/PIRM-SR.html).

Followings are the performance comparison evaluated on the [BSD100](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) dataset.

Method | PSNR (dB) (↓) | SSIM | Perceptual Index
------------ | :---: | :---: | :---:
EDSR | 27.796 | 0.744 | 5.326
MDSR | 27.771 | 0.743 | 5.424
EUSR | 27.674 | 0.740 | 5.307
SRResNet-MSE | 27.601 | 0.737 | 5.217
**4PP-EUSR (PIRM Challenge)** | 26.569 | 0.688 | 2.683
SRResNet-VGG22 | 26.322 | 0.694 | 5.183
SRGAN-MSE | 25.981 | 0.643 | 2.802
Bicubic interpolation | 25.957 | 0.669 | 6.995
SRGAN-VGG22 | 25.697 | 0.660 | 2.631
SRGAN-VGG54 | 25.176 | 0.641 | 2.351
CX | 24.581 | 0.644 | 2.250

Please cite following papers when you use the code, pre-trained models, or results:
- J.-H. Choi, J.-H. Kim, M. Cheon, J.-S. Lee: Deep learning-based image super-resolution considering quantitative and perceptual quality. arXiv:xxxx.xxxxx (2018) (will be available soon)
- J.-H. Kim, J.-S. Lee: [Deep residual network with enhanced upscaling module for super-resolution](http://openaccess.thecvf.com/content_cvpr_2018_workshops/w13/html/Kim_Deep_Residual_Network_CVPR_2018_paper.html). In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops (2018)

## Dependencies
- Python 3.6+
- TensorFlow 1.8+

## Test pre-trained models
Generating upscaled images from the trained models can be done by `test/test.py`.
Following are the brief instructions.

1. Download and copy the trained model available in [Downloads](#downloads) section to the `test/` folder.
2. Place the low-resolution images (PNG only) to the `test/LR/` folder.
3. Run `python test.py --model_name [model file name]`. For example, if you downloaded the PIRM Challenge version of our pre-trained model, run `python test.py --model_name 4pp_eusr_pirm.pb`.
4. The upscaled images will be available on the `test/SR/` folder.

Please run `python test.py --help` for more information.

## Training 4PP-EUSR
The training code is not yet available.
Please stay tuned for further updates. :)

## Downloads

Pre-trained models:
- PIRM Challenge version: [4pp_eusr_pirm.pb](http://mcml.yonsei.ac.kr/files/4pp_eusr/4pp_eusr_pirm.pb)
- Paper version: [4pp_eusr_paper.pb](http://mcml.yonsei.ac.kr/files/4pp_eusr/4pp_eusr_paper.pb)

Results (Set5, Set14, BSD100, PIRM):
- PIRM Challenge version: [4pp_eusr_results_pirm.zip](http://mcml.yonsei.ac.kr/files/4pp_eusr/4pp_eusr_results_pirm.zip)
- Paper version: [4pp_eusr_results_paper.zip](http://mcml.yonsei.ac.kr/files/4pp_eusr/4pp_eusr_results_paper.zip)
