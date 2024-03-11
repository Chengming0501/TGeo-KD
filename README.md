# TGeo-KD
This repository contains the preprint and code for the paper "Less or More From Teacher: Exploiting Trilateral Geometry For Knowledge Distillation" (ICLR 2024).
## Installation
```
pip install torch
````
## Reproducing Results
Data preparation:

You can download the dataset under the `data` folder through the link provided in our paper available [here](https://arxiv.org/abs/2312.15112). For HIL and Criteo datasets, you may need to perform data normalization and random oversampling as the steps of the data pre-processing. 

Pre-train the teacher network:
````
python vanilla_kd.py
````

To run all baselines:
```
python baseline_ADA.py
python baseline_ANL.py
python baseline_RW.py
python baseline_WLS.py
````
To run TGeo-KD:
```
python fusion_ratio_bilevel.py
````
Hyperparameter settings:

The settings of main hyperparameters can be found in our Appendix available [here](https://arxiv.org/abs/2312.15112). Please note that optimal hyperparameters may vary due to differences in the dataset partition.

## Reference
Please cite our paper if you use the core code of TGeo-KD.

```
@inproceedings{hu2023less,
  title={Less or More From Teacher: Exploiting Trilateral Geometry For Knowledge Distillation},
  author={Hu, Chengming and Wu, Haolun and Li, Xuan and Ma, Chen and Chen, Xi and Wang, Boyu and Yan, Jun and Liu, Xue},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2023}
}
```
