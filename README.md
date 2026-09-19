# DWCF: Dominance-aware Weak-Modality Compensation Framework

This repository provides the core implementation of **DWCF**, a framework for alleviating modality imbalance in multimodal learning.

## Method overview

DWCF contains three complementary components:

1. **Counterfactual Modality Dominance Regulation (CMDR)** estimates modality dominance through counterfactual masking and regulates optimization accordingly.
2. **Progressive Hard-Sample Focusing (PHSF)** gradually emphasizes difficult samples during training.
3. **Distillation-Guided Weak-Modality Compensation (DWMC)** uses auxiliary specialist heads and fused predictions to strengthen weak unimodal branches.

The current code supports experiments on **CREMA-D**, **Kinetics-Sounds**, and **CMU-MOSI**.

## Repository structure

```text
DWCF/
├── main.py                         # CREMA-D and Kinetics-Sounds training
├── main_trimodal_mosi_darf_v3.py   # CMU-MOSI tri-modal training
├── KSDataset.py                    # Kinetics-Sounds loader
├── dataset/                        # CREMA-D and CMU-MOSI loaders
├── models/                         # backbones, fusion modules and auxiliary heads
└── utils/                          # CMDR, checkpoint goals and balancing utilities
```

## Installation

Python 3.8+ is recommended.

```bash
pip install -r requirements.txt
```

## Data preparation

Datasets are not redistributed. Download them from their official sources and organize the processed files as follows, or provide equivalent paths through the command-line arguments.

```text
data/
├── CREMA-D/
│   └── CREMAD/
│       ├── train.csv
│       ├── test.csv
│       ├── AudioWAV/
│       └── Image-03-FPS/
├── KineticSound/
│   ├── my_train_fixed.txt
│   ├── my_test_fixed.txt
│   ├── train_spec/
│   ├── test_spec/
│   ├── train-videos/train-set-img/Image-01-FPS/
│   └── test-videos/test-set-img/Image-01-FPS/
└── CMU-MOSI/
    └── Processed/
        └── unaligned_50.pkl
```

## Training

### CREMA-D

```bash
python main.py --train --dataset CREMAD \
  --cremad_root ./data/CREMA-D \
  --audio_path ./data/CREMA-D/CREMAD/AudioWAV \
  --visual_path ./data/CREMA-D/CREMAD \
  --use_cmob --use_contrastive --use_sample_weighting \
  --use_branch_specialist --wbsr_loss_scale 2.0
```

### Kinetics-Sounds

```bash
python main.py --train --dataset KineticSound \
  --ks_data_root ./data/KineticSound \
  --use_cmob --use_contrastive --use_sample_weighting \
  --use_branch_specialist --wbsr_loss_scale 2.0
```

### CMU-MOSI

```bash
python main_trimodal_mosi_darf_v3.py --train --dataset MOSI \
  --data_path ./data/CMU-MOSI/Processed \
  --dataset_name unaligned_50
```

Use `python main.py --help` or `python main_trimodal_mosi_darf_v3.py --help` for the complete configuration list.

## Reproducibility notes

- Dataset files, checkpoints and training logs are intentionally excluded from the repository.
- Default paths are portable relative paths; override them for your local environment.
- Set `--random_seed` explicitly when comparing methods.
- Evaluation settings such as frame sampling and multi-view evaluation are exposed as command-line options.

## Citation

Citation information will be added when the paper becomes available.
