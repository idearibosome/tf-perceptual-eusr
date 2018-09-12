# 4PP-EUSR
Four-pass perceptual super-resolution with enhanced upscaling


## Introduction
This repository contains a TensorFlow-based implementation of 4PP-EUSR, which considers both the quantitative and perceptual quality of the upscaled images.
Our method won the 2nd place for Region 2 in the [PIRM Challenge on Perceptual Super Resolution at ECCV 2018](https://www.pirm2018.org/PIRM-SR.html).

![BSD100 - 37073](figures/bsd100_37073.png)
※ The perceptual index is calculated by "0.5 * ((10 - [Ma](https://sites.google.com/site/chaoma99/sr-metric)) + [NIQE](https://doi.org/10.1109/LSP.2012.2227726))", which is used in the [PIRM Challenge](https://www.pirm2018.org/PIRM-SR.html).


## Dependencies
- Python 3.6+
- TensorFlow 1.8+

## Test pre-trained models
Generating upscaled images from the trained models can be done by `test/test.py`.
Following are the brief instructions.

1. Download and copy the trained model available in [Downloads](#downloads) section to the `test/` folder.
2. Place the low-resolution images (PNG only) to the `test/LR` folder.
3. Run `python test.py --model_name [model file name]`. For example, if you downloaded the PIRM Challenge version of our pre-trained model, run `python test.py --model_name 4pp_eusr_pirm.pb`.
4. The upscaled images will be available on the `test/SR` folder.

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
