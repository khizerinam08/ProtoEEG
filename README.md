# ProtoEEG

Official repository for the paper:

**"This EEG Looks Like These EEGs: Interpretable Interictal Epileptiform Discharge Detection With ProtoEEG-kNN"**

## Installation

```bash
# Clone the repository
git clone https://github.com/DennisTang2000/ProtoEEG.git
cd ProtoEEG

# Install dependencies
pip install -r requirements.txt
```

## Data

The dataset used in this paper is available at: https://github.com/bdsp-core/SpikeNet2

We include sample data in the data directory.

**Note:** The EEG data storage format is based on an existing codebase. We recommend building your own dataloaders to suit your specific data format and needs.

## Pre-trained Model

The fully trained model is available here:
https://drive.google.com/drive/folders/13BC4A71n91kT7yRBfdXatQ7tdoD7MFXd?usp=drive_link

## Usage

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
python eval_eegprotopnet.py -path ./models/trained_model.pth -topk 4
```

## Training

The best hyperparameters for training are provided in:
```
training/sweeps/MICCAI/best_normal.yaml
```

## Contact

For questions or feedback, please open an issue on GitHub or email dt161@duke.edu