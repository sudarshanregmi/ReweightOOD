# ReweightOOD: Loss Reweighting for Distance-based OOD Detection 
This codebase provides a Pytorch implementation of:

>**ReweightOOD: Loss Reweighting for Distance-based OOD Detection.**  
[![ReweightOOD](https://img.shields.io/badge/CVPRW'24-ReweightOOD-f4d5b3?style=for-the-badge)](https://openaccess.thecvf.com/content/CVPR2024W/TCV2024/papers/Regmi_ReweightOOD_Loss_Reweighting_for_Distance-based_OOD_Detection_CVPRW_2024_paper.pdf)  
Sudarshan Regmi, Bibek Panthi, Yifei Ming, Prashnna K. Gyawali, Danail Stoyanov, Binod Bhattarai

## Abstract
Out-of-Distribution (OOD) detection is crucial for ensuring safety and reliability of neural networks in critical applications. Distance-based OOD detection is based on the assumption that OOD samples are mapped far from In-Distribution (ID) clusters in embedding space. A recent approach for obtaining OOD-detection-friendly embedding space has been contrastive optimization of pulling similar pairs and pushing apart dissimilar pairs. It assigns equal significance to all similarity instances with the implicit objective of maximizing the mean proximity between samples with their corresponding hypothetical class centroids. However, the emphasis should be directed towards reducing the Minimum Enclosing Sphere (MES) for each class and achieving higher inter-class dispersion to effectively mitigate the potential for ID-OOD overlap. Optimizing low-signal dissimilar pairs might potentially act against achieving maximal inter-class dispersion while less-optimized similar pairs prevent achieving smaller MES. Based on this, we propose a reweighting scheme ReweightOOD, that adopts the similarity optimization which prioritizes the optimization of less-optimized contrasting pairs while assigning lower importance to already well-optimized contrasting pairs. Such a reweighting scheme serves to minimize the MES for each class while achieving maximal inter-class dispersion. Experimental results on a challenging CIFAR100 benchmark using ResNet-18 network demonstrate that ReweightOOD outperforms supervised contrastive loss by a whopping 38% in the average FPR metric. In various classification datasets, our method provides a promising solution for enhancing OOD detection capabilities in neural networks.

### ReweightOOD highlights in OpenOODv1.5 [leaderboard](https://zjysteven.github.io/OpenOOD/) [Updated: June 2024]

➤ **ranks 2nd** in the *far-OOD AUROC* metric on CIFAR10 datasets.  
➤ **ranks 1st** in the *far-OOD AUROC* metric on CIFAR100 datasets.  
➤ **ranks 5th** in the *far-OOD AUROC* metric on ImageNet200 datasets (OOD).  
➤ **ranks 2nd** in the *far-OOD AUROC* metric on ImageNet200 datasets (FSOOD).  
➤ **ranks 9th** in the *far-OOD AUROC* metric on ImageNet-1k datasets (OOD).  
➤ **ranks 6th** in the *far-OOD AUROC* metric on ImageNet-1k datasets (FSOOD).  

<span>Check other works:</span>

<a href="https://github.com/sudarshanregmi/ascood"><img src="https://img.shields.io/badge/arXiv'24-ASCOOD-fdd7e6?style=for-the-badge" alt="ascood" style="margin-right: 10px;"></a> <br>
<a href="https://github.com/sudarshanregmi/t2fnorm"><img src="https://img.shields.io/badge/CVPRW'24-ReweightOOD-fdd7e6?style=for-the-badge" alt="t2fnorm" style="margin-right: 10px;"></a>

### Follow [OpenOOD](https://github.com/Jingkang50/OpenOOD) official instruction to complete the setup.
```
pip install git+https://github.com/Jingkang50/OpenOOD
```

### Method Illustration
<p align="center">
  <img width="800" src="ReweightOOD.png">
</p>

### Example Scripts for Training and Inference
Use the following scripts for training and inferencing the ReweightOOD model on different datasets:

- **CIFAR-10:**
  ```bash
  bash scripts/ood/reweightood/cifar10_train_reweightood.sh
  bash scripts/ood/reweightood/cifar10_test_reweightood.sh
  ```
- **CIFAR-100:**
  ```bash
  bash scripts/ood/reweightood/cifar100_train_reweightood.sh
  bash scripts/ood/reweightood/cifar100_test_reweightood.sh
  ```
- **ImageNet-200:**
  ```bash
  bash scripts/ood/reweightood/imagenet200_train_reweightood.sh
  bash scripts/ood/reweightood/imagenet200_test_reweightood.sh
  ```
- **ImageNet-1k:**
  ```bash
  bash scripts/ood/reweightood/imagenet_train_reweightood.sh
  bash scripts/ood/reweightood/imagenet_test_reweightood.sh
  ```

### Results

- CIFAR-10 (OOD):

|   datasets   | FPR@95       | AUROC        | AUPR_IN      | AUPR_OUT     |
|:-------------|:-------------|:-------------|:-------------|:-------------|
| cifar100     | 37.68 ± 0.97 | 89.64 ± 0.08 | 90.33 ± 0.11 | 88.55 ± 0.09 |
| tin          | 23.79 ± 0.86 | 94.09 ± 0.30 | 95.43 ± 0.24 | 92.31 ± 0.44 |
| nearood      | 30.74 ± 0.79 | 91.86 ± 0.16 | 92.88 ± 0.17 | 90.43 ± 0.19 |
| mnist        | 6.33 ± 0.71  | 98.40 ± 0.19 | 94.91 ± 0.52 | 99.73 ± 0.03 |
| svhn         | 0.32 ± 0.03  | 99.87 ± 0.01 | 99.65 ± 0.03 | 99.96 ± 0.00 |
| texture      | 10.48 ± 0.84 | 98.09 ± 0.13 | 98.83 ± 0.08 | 97.12 ± 0.21 |
| places365    | 17.05 ± 1.15 | 95.82 ± 0.31 | 90.46 ± 0.62 | 98.64 ± 0.12 |
| farood       | 8.54 ± 0.35  | 98.05 ± 0.10 | 95.96 ± 0.26 | 98.86 ± 0.05 |

- CIFAR-100 (OOD):

|   datasets   | FPR@95       | AUROC        | AUPR_IN      | AUPR_OUT     |
|:-------------|:-------------|:-------------|:-------------|:-------------|
| cifar10      | 86.11 ± 1.33 | 63.31 ± 0.76 | 60.62 ± 1.25 | 62.48 ± 0.73 |
| tin          | 64.70 ± 0.48 | 79.24 ± 0.05 | 84.21 ± 0.13 | 70.19 ± 0.11 |
| nearood      | 75.41 ± 0.86 | 71.27 ± 0.41 | 72.42 ± 0.69 | 66.34 ± 0.40 |
| mnist        | 35.22 ± 1.80 | 89.90 ± 1.55 | 73.43 ± 1.53 | 98.19 ± 0.39 |
| svhn         | 8.83 ± 1.01  | 98.44 ± 0.19 | 96.33 ± 0.39 | 99.44 ± 0.07 |
| texture      | 32.64 ± 2.98 | 93.11 ± 0.56 | 95.25 ± 0.39 | 90.17 ± 0.72 |
| places365    | 54.02 ± 2.00 | 83.04 ± 0.51 | 65.02 ± 1.59 | 93.35 ± 0.14 |
| farood       | 32.68 ± 0.49 | 91.12 ± 0.33 | 82.51 ± 0.46 | 95.29 ± 0.06 |

- ImageNet-200 (OOD):

|   datasets   | FPR@95       | AUROC        | AUPR_IN      | AUPR_OUT     |
|:-------------|:-------------|:-------------|:-------------|:-------------|
| ssb_hard     | 68.20 ± 0.36 | 76.34 ± 0.35 | 42.35 ± 0.28 | 93.45 ± 0.13 |
| ninco        | 48.54 ± 0.89 | 84.36 ± 0.15 | 89.99 ± 0.13 | 75.66 ± 0.24 |
| nearood      | 58.37 ± 0.57 | 80.35 ± 0.22 | 66.17 ± 0.17 | 84.55 ± 0.15 |
| inaturalist  | 24.24 ± 0.64 | 93.93 ± 0.17 | 94.38 ± 0.17 | 93.96 ± 0.15 |
| textures     | 14.54 ± 0.44 | 97.33 ± 0.14 | 98.35 ± 0.08 | 95.95 ± 0.21 |
| openimage_o  | 31.61 ± 0.25 | 90.72 ± 0.09 | 88.30 ± 0.03 | 93.56 ± 0.10 |
| farood       | 23.46 ± 0.31 | 93.99 ± 0.08 | 93.68 ± 0.07 | 94.49 ± 0.12 |

- ImageNet-200 (FSOOD):

|   datasets   | FPR@95       | AUROC        | AUPR_IN      | AUPR_OUT     |
|:-------------|:-------------|:-------------|:-------------|:-------------|
| ssb_hard     | 89.85 ± 0.12 | 46.11 ± 0.19 | 48.35 ± 0.13 | 49.29 ± 0.20 |
| ninco        | 81.63 ± 0.28 | 57.24 ± 0.42 | 91.51 ± 0.08 | 14.80 ± 0.29 |
| nearood      | 85.74 ± 0.18 | 51.68 ± 0.18 | 69.93 ± 0.06 | 32.04 ± 0.18 |
| inaturalist  | 66.34 ± 0.41 | 77.03 ± 0.35 | 93.39 ± 0.09 | 47.05 ± 0.81 |
| textures     | 52.51 ± 1.05 | 89.20 ± 0.53 | 98.31 ± 0.08 | 62.10 ± 1.82 |
| openimage_o  | 72.40 ± 0.14 | 68.68 ± 0.70 | 86.39 ± 0.26 | 45.00 ± 0.69 |
| farood       | 63.75 ± 0.36 | 78.31 ± 0.46 | 92.70 ± 0.11 | 51.38 ± 0.92 |

- ImageNet-1k (OOD):

|   datasets   |   FPR@95 |   AUROC |   AUPR_IN |   AUPR_OUT |
|:-------------|---------:|--------:|----------:|-----------:|
| ssb_hard     |    86.3  |   61.1  |     59.77 |      60.37 |
| ninco        |    58.75 |   80.07 |     96.78 |      32.58 |
| nearood      |    72.53 |   70.58 |     78.27 |      46.48 |
| inaturalist  |    21.43 |   93.55 |     98.46 |      72.67 |
| textures     |     5.19 |   98.84 |     99.85 |      93.85 |
| openimage_o  |    33.85 |   92.05 |     96.97 |      80.57 |
| farood       |    20.16 |   94.82 |     98.43 |      82.36 |

- ImageNet-1k (FSOOD):

|   datasets   |   FPR@95 |   AUROC |   AUPR_IN |   AUPR_OUT |
|:-------------|---------:|--------:|----------:|-----------:|
| ssb_hard     |    92.16 |   43.82 |     65.47 |      28.58 |
| ninco        |    74.45 |   62.5  |     96.77 |       7.24 |
| nearood      |    83.3  |   53.16 |     81.12 |      17.91 |
| inaturalist  |    44.7  |   80.36 |     97.63 |      21.29 |
| textures     |    21.38 |   95.27 |     99.71 |      60.32 |
| openimage_o  |    56.13 |   79.4  |     95.96 |      32.85 |
| farood       |    40.74 |   85.01 |     97.76 |      38.16 |

### Pre-trained checkpoints
Pre-trained models are available in the given links:
- CIFAR-10 [[Google Drive]](https://drive.google.com/file/d/1rGcwveiqhz2a0UIbBgetG3fKSLDv_ZH7/view?usp=sharing): ResNet-18 backbone trained on CIFAR-10 datasets with ReweightOOD objective across 3 trials.
- CIFAR-100 [[Google Drive]](https://drive.google.com/file/d/13a3fBplLfIJ2VxXCNlsyXXP9Wv3rm1e7/view?usp=sharing): ResNet-18 backbone trained on CIFAR-100 datasets with ReweightOOD objective across 3 trials.
- ImageNet-200 [[Google Drive]](https://drive.google.com/file/d/1lDZe16iye7c1llXosSFM5P5TGYHZp08B/view?usp=sharing): ResNet-18 backbone trained on ImageNet-200 datasets with ReweightOOD objective across 3 trials.
- ImageNet-1k [[Google Drive]](https://drive.google.com/file/d/1DEq7Tppr_7vX-246zXeAmSNl4UBX-otp/view?usp=drive_link): ResNet-50 backbone fine-tuned on ImageNet-1k datasets with ReweightOOD objective across 1 trial.

### Please consider citing our work if you find it useful.
```
@InProceedings{Regmi_2024_CVPR_1,
    author    = {Regmi, Sudarshan and Panthi, Bibek and Ming, Yifei and Gyawali, Prashnna K and Stoyanov, Danail and Bhattarai, Binod},
    title     = {ReweightOOD: Loss Reweighting for Distance-based OOD Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024},
    pages     = {131-141}
}
```
### Acknowledgment
This codebase builds upon [OpenOOD](https://github.com/Jingkang50/OpenOOD).
