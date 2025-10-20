# ProtoEEG

Repo for MICCAI 2025 paper:

**"This EEG Looks Like These EEGs: Interpretable Interictal Epileptiform Discharge Detection With ProtoEEG-kNN"**

This code was built on a pre-release version of [ProtoPNeXt](https://arxiv.org/abs/2406.14675)

## Installation

```bash
# Clone the repository
git clone https://github.com/DennisTang2000/ProtoEEG.git
cd ProtoEEG

# Create a venv (suggested) and install dependencies
pip install -r requirements.txt
```

## Data

The dataset used in this paper is available at: https://github.com/bdsp-core/SpikeNet2. As of 10/17, the team is still in the process of fixing the data availability, please contact them for questions related to data.

We include sample data (n=10) in the sample_data directory.

**Note:** The EEG data storage format is based on an existing codebase. We recommend building your own dataloaders to suit your specific data format and needs.

## Pre-trained Model

The fully trained model is available here:
https://drive.google.com/drive/folders/13BC4A71n91kT7yRBfdXatQ7tdoD7MFXd?usp=drive_link


**Note:** To use our full model, you will need access to the entire training set (for the kNN-replacement step). However, users can specify their own dataset in `protopnet/eval_utils.py` and use our trained model with any dataset.

### Setup Instructions

1. **Download the model** from the link above and place `trained_model.pth` in the root directory of this repository (same level as the `protopnet/` folder). This is because `torch.load()` needs to import the `protopnet` module to reconstruct the model architecture. The model can also be placed in a subdirectory at the same level as `protopnet/`, then load it with `torch.load("subdir/trained_model.pth")`.

2. See `demo_inference.ipynb` for an example of how to use the provided model for inference.


### Create channel-wise weights
```bash
python create_spikenet_labels.py
```

### Visualize model predictions
```bash
python viz_local_analysis.py
```

We include 10 example visualizations in the `viz` directory.

### Evaluate the model
```bash
python eval_eegprotopnet.py -path ./models/trained_model.pth -topk 10
```

## Training

The best hyperparameters for training are provided in:
```
training/sweeps/MICCAI/best_normal.yaml
```

## Contact

For questions or feedback, please open an issue on GitHub or email dt161@duke.edu