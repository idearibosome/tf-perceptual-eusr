# 4PP-EUSR
Four-pass perceptual super-resolution with enhanced upscaling

## Introduction
This repository contains a TensorFlow-based implementation of 4PP-EUSR, which considers both the quantitative and perceptual quality of the upscaled images.
Our method won the 2nd place for Region 2 in the [PIRM Challenge on Perceptual Super Resolution at ECCV 2018](https://www.pirm2018.org/PIRM-SR.html).

![BSD100 - 37073](figures/bsd100_37073.png)
※ The perceptual index is calculated by "0.5 * ((10 - [Ma](https://sites.google.com/site/chaoma99/sr-metric)) + [NIQE](https://doi.org/10.1109/LSP.2012.2227726))", which is used in the [PIRM Challenge](https://www.pirm2018.org/PIRM-SR.html).


## Downloads

Pre-trained models:
- PIRM Challenge version: [4pp_eusr_pirm.pb](http://mcml.yonsei.ac.kr/files/4pp_eusr/4pp_eusr_pirm.pb)
- Paper version: [4pp_eusr_paper.pb](http://mcml.yonsei.ac.kr/files/4pp_eusr/4pp_eusr_paper.pb)

Results (Set5, Set14, BSD100, PIRM):
- PIRM Challenge version: [4pp_eusr_results_pirm.zip](http://mcml.yonsei.ac.kr/files/4pp_eusr/4pp_eusr_results_pirm.zip)
- Paper version: [4pp_eusr_results_paper.zip](http://mcml.yonsei.ac.kr/files/4pp_eusr/4pp_eusr_results_paper.zip)
