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

