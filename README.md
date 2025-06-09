# Image DeMoiréing Using Dual Camera Fusion on Mobile Phones
 (ICME 2025)


## Overview


 When shooting electronic screens, Moiré patterns usually appear in captured images, which seriously affects the image quality. Existing image deMoiréing methods face great challenges in removing large and heavy Moiré. To address the issue, we propose to utilize Dual Camera fusion for Image DeMoiréing (DCID), i.e., using the ultra-wide-angle (UW) image to assist the Moiré removal of wide-angle (W) image. This is inspired by two motivations: (1) the two lenses are commonly equipped with modern smartphones, (2) the UW image generally can provide normal colors and textures when Moiré exists in the W image mainly due to their different focal lengths. In particular, we propose an efffcient DCID method, where a lightweight UW image encoder is integrated into an existing deMoiréing network and a fast two-stage image alignment manner is present. Moreover, we construct a large-scale real-world dataset with diverse mobile phones and monitors, containing about 9,000 samples. We hope our investigation could inspire more future research in this more practical setting in image demoiréing.



##  Datasets 
Download Links:

1. Baidu disk: [here](https://pan.baidu.com/s/1IAFmdE4Mkh9ATi2DRspcVQ?pwd=9aab)  (Code: 9aab )

2. HuggingFace: [here](https://huggingface.co/datasets/Mrduckk/DCID)

## Environments

```
conda env dcid -f environment.yaml
conda activate dcid
```
## Train

The configuration file (config) uses the parameter TRAIN_WORD to denote three different datasets, with each value corresponding to a specific dataset as follows:
xiaomi: Represents the xiaomi dataset.
iphone: Represents the iphone dataset.
p40: Represents the huawei dataset.

The parameter align determines whether to use pre-generated alignment results from Keypoint Matching based Alignment(KMA) during training:
false: The network will perform the first-stage alignment internally during the training process.
true: The network will directly use the precomputed alignment results from the align folder.
Specifically, it will use the file named w_align.jpg in the align directory for alignment, skipping the KMA during training.

```
python train.py --config ./config/dcid_config.yaml
```

## Test
pre-trained models on our three datasets: [here](https://pan.baidu.com/s/1mbMAksiBkRshzIPK7coDmg?pwd=9mqy) (Code: 9mqy )

```
python test.py --config ./config/test_dcid_config.yaml
```

## Acknowledgement
This project is built based on [UHDM](https://github.com/CVMI-Lab/UHDM?tab=readme-ov-file).

## Citation
If you find this work or code useful for your research, please cite:
```


```
