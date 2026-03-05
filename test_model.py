"""
Quick test script to verify the model loads and runs inference correctly.
Uses the train labels as eval labels just for verification purposes.
"""
from sklearn.metrics import roc_auc_score
from protopnet.knn_models import ProtoEEGkNN
from protopnet.eval_utils import bootstrap_metrics_ci
from protopnet.spikenet_helpers import EEG_ConcatDataset
import torch
import numpy as np

# --- CONFIG ---
MODEL_PATH = "/home/muhammad-adeel-ajmal-khan/Documents/repos/ProtoEEG/trained_model.pth"
DATA_FOLDER = "/home/muhammad-adeel-ajmal-khan/Documents/snd/real_npy"
LABELS_FILE = "/home/muhammad-adeel-ajmal-khan/Documents/repos/ProtoEEG/sn2_data/organized_data/sn2_train_labels.npy"

# 1. Load model (KNN version)
print(f"Loading model from: {MODEL_PATH}")
model = ProtoEEGkNN(MODEL_PATH, topk=10)

sm = torch.nn.Softmax(dim=0)
importance_stats = sm(model.base_model.prototype_layer.importance_by_statistic)
print("Model importance stats (latent, range, var, FFT): ", importance_stats)

# --- CRITICAL FIX: Reload the new spikenet weights ---
# The saved model has an old/empty dict. We must inject the new one we just made.
print("Reloading spikenet weights from model_feats/spikenet_labels.pth...")
spikenet_weights = torch.load("model_feats/spikenet_labels.pth")
model.base_model.prototype_layer.spikenet_weight_dict = spikenet_weights
print(f"Loaded {len(spikenet_weights)} spikenet weights.")

# 2. Create test loader using train labels (just for verification)
customDataSet_kw_args = {
    "eeg_data": {
        "train": DATA_FOLDER,
        "train_push": DATA_FOLDER,
        "eval": DATA_FOLDER,
    },
    "labels": {
        "train": LABELS_FILE,
        "train_push": LABELS_FILE,
        "eval": LABELS_FILE,
    },
    "threshold": 0.5,
    "train_transform": None,
    "push_transform": "spikenet_helpers.eeg_crop spikenet_helpers.spikenet_transform spikenet_helpers.extremes_remover spikenet_helpers.normalizer",
    "eval_transform": "spikenet_helpers.eeg_crop spikenet_helpers.spikenet_transform spikenet_helpers.extremes_remover spikenet_helpers.normalizer",
}

test_dataset = EEG_ConcatDataset(mode="eval", **customDataSet_kw_args)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# 3. Run inference
print("Starting inference...")
y_true = np.array([])
y_pred = np.array([])

for batch_idx, sample in enumerate(test_loader):
    with torch.no_grad():
        eeg = sample["img"]
        input_ids = sample["sample_id"]
        target = sample["target"]
        y_true = np.concatenate((y_true, target))

        output_dict = model.forward(eeg, input_ids)
        prediction = output_dict["prediction"]
        y_pred = np.concatenate((y_pred, prediction))

    if batch_idx % 10 == 0:
        print(f"  Batch {batch_idx}/{len(test_loader)}...")

# 4. Compute metrics
print("\nComputing metrics...")
results = bootstrap_metrics_ci(y_true, y_pred)

print("=" * 50)
print("Bootstrap Metrics with 95% Confidence Intervals")
print("=" * 50)
print(f"R² Score:    {results['r2']:.4f} (CI: [{results['r2_ci'][0]:.4f}, {results['r2_ci'][1]:.4f}])")
print(f"Accuracy:    {results['accuracy']:.4f} (CI: [{results['accuracy_ci'][0]:.4f}, {results['accuracy_ci'][1]:.4f}])")
print(f"AUROC:       {results['auroc']:.4f} (CI: [{results['auroc_ci'][0]:.4f}, {results['auroc_ci'][1]:.4f}])")
print("=" * 50)
print(f"\nTotal samples evaluated: {len(y_true)}")
