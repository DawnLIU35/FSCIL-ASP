# FSCIL-ASP Official Implementation
This codebase contains the official Python implementation of [Few-shot Class Incremental Learning with Attention-Aware Self-Adaptive Prompt](https://arxiv.org/pdf/2403.09857) (ECCV2024)

## Introduction
ASP is a novel few-shot class incremental learning (FSCIL) algorithm which utilizes prompt tuning with a Vision Transformer backbone. ASP encourages task-invariant prompts to capture shared knowledge by reducing specific information from the attention aspect. Additionally, self-adaptive task-specific prompts in ASP provide specific information and transfer knowledge from old classes to new classes with an Information Bottleneck learning objective.
![](https://github.com/DawnLIU35/FSCIL-ASP/blob/main/fig/alg.png)

### Performance
ASP consistently outperforms traditional FSCIL works using ResNet, multi-modal FSCIL works using CLIP, and prompt-based CIL works using ViT.
![](https://github.com/DawnLIU35/FSCIL-ASP/blob/main/fig/main_result.png)

## Instructions on running ASP

### Environment setup
Clone this GitHub repository and run:
```
pip install -r requirements.txt
```

### Dataset preparation
Download the dataset and put them in folder `./data`
* **CIFAR100**: will be automatically downloaded by the code.
* **CUB200**: Google Drive: [link](https://drive.google.com/file/d/1jx0ICqvgaXyfWUVLTv6St0_b6p7D0Hpm/view?usp=sharing)
* **ImageNet-R**: Google Drive: [link](https://drive.google.com/file/d/1R4bRjYXnbRWje6hw_YPdsKr1HuojXzqO/view?usp=sharing)

### Run experiment
Experiment on CIFAR100 dataset:
```
python main.py --config=./exps/cifar.json
```
Experiment on CUB200 dataset:
```
python main.py --config=./exps/cub.json
```
Experiment on ImageNet-R dataset:
```
python main.py --config=./exps/inr.json
```

## Citation
```
@article{liu2024few,
  title={Few-Shot Class Incremental Learning with Attention-Aware Self-Adaptive Prompt},
  author={Liu, Chenxi and Wang, Zhenyi and Xiong, Tianyi and Chen, Ruibo and Wu, Yihan and Guo, Junfeng and Huang, Heng},
  journal={arXiv preprint arXiv:2403.09857},
  year={2024}
}
```

