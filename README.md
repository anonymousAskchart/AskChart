# AskChart: Universal Chart Understanding through Textual Enhancement

## Contents
- [Install](#install)
- [Model Weights](#model-weights)
- [Evaluation Code](#evaluation-code)
- [Training Code ](#training-code)
- [Dataset](#dataset)

## Install
1. Clone this repository and navigate to AskChart folder
```bash
cd AskChart
```

2. Install Package
```Shell
conda create -n askchart python=3.10 -y
conda activate askchart
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases
```Shell
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## Model Weights
The model weights are on the to-do list. You can also load the stage II pre-trained model of Moe-LLaVA(https://github.com/PKU-YuanGroup/MoE-LLaVA) as the initial weights and retrain the AskChart model.

## Evaluation Code
The evaluation instruction is in [EVAL.md](docs/EVAL.md).

## Training Code
The training process includes three stages.

- Stage 1 pretraining script: [pretrain.sh](https://github.com/anonymousAskchart/AskChart/tree/main/scripts/v1/phi2/pretrain.sh). 
- Stage 2 tuning script: [finetune.sh](https://github.com/anonymousAskchart/AskChart/tree/main/scripts/v1/phi2/finetune.sh).
- Stage 3 moe-tuning script: [finetune_moe.sh](https://github.com/anonymousAskchart/AskChart/tree/main/scripts/v1/phi2/finetune_moe.sh).

## Dataset
Due to the large and complex nature of the dataset, the complete data is still being organized. Below is a subset of the data for preview purposes.

The subset of [ChartBase](https://drive.google.com/file/d/1IKL8-DMTbZko1z5TBNJaxknJ1B3pmyL-/view?usp=sharing) and [annotation-file](https://drive.google.com/file/d/1oD7VliLDGfJqtXPrrlJYsYeJvWzsmZEP/view?usp=sharing)

## 👍 Acknowledgement
* [LLaVA](https://github.com/haotian-liu/LLaVA) and [Moe-LLaVA](https://github.com/PKU-YuanGroup/MoE-LLaVA) The codebase we built upon and they are excellent multi-modal assistants.