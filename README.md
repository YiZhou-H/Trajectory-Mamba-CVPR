# Trajectory Mamba: Efficient Attention-Mamba Forecasting Model Based on Selective SSM

This repository contains the official implementation of *Trajectory Mamba: Efficient Attention-Mamba Forecasting Model Based on Selective SSM*. Trajectory Mamba is an efficient motion forecasting framework for autonomous driving systems. It leverages a selective state-space model (Selective SSM) to redesign the self-attention mechanism in an encoder-decoder architecture, reducing computational complexity from quadratic to linear. A joint polyline encoding strategy is also introduced to improve prediction accuracy. Experiments on the Argoverse 1 and Argoverse 2 datasets demonstrate state-of-the-art performance with significantly reduced FLOPs and model parameters.

---

## Quick Start

### 1. Installation

We recommend using an Anaconda virtual environment. Run the following commands:

```bash
conda env create -f environment.yaml
conda activate trajectory_mamba
pip install -r requirements.txt
```

### 2. Data Preparation

Organize your dataset as follows:

```
data/
├── Argoverse1/
│   ├── images/
│   └── annotations.json
└── Argoverse2/
    ├── images/
    └── annotations.json
```

### 3. Download Checkpoints

Download the pretrained models from [Pretrained Model Link] and place them in the `./checkpoints` directory:

```
checkpoints/
├── argoverse1_ckpt.pth
└── argoverse2_ckpt.pth
```

### 4. Testing & Evaluation

Run the evaluation script:

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py --config configs/main_train.yml --exp_id "main_train" --distributed
```

*Note: Evaluation is recommended on an NVIDIA RTX 4090 GPU (24GB VRAM). For multi-GPU usage, specify the GPU IDs accordingly.*

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{huang2024trajectory,
  title={Trajectory Mamba: Efficient Attention-Mamba Forecasting Model Based on Selective SSM},
  author={Huang, Yizhou and Cheng, Yihua and Wang, Kezhi},
  journal={arXiv preprint arXiv:2503.10898},
  year={2024}
}
```

---

## License

This project is licensed under the MIT License.# Trajectory-Mamba
