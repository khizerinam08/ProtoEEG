from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    r2_score,
    precision_recall_curve,
    auc,
)
from protopnet.spikenet_helpers import EEG_ConcatDataset
import torch
import os
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.utils import resample
import numpy as np
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score
from sklearn.utils import resample


def bootstrap_metrics_ci(y_true, y_pred, n_iterations=10000):
    """
    Calculate original R², accuracy,and AUROC with 95% confidence intervals via bootstrapping.

    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    n_iterations : int, default=10000
        Number of bootstrap iterations

    Returns:
    --------
    dict : Dictionary containing original metrics and their 95% confidence intervals
    """
    # Ensure numpy arrays
    y_true = np.array(y_true/8)
    y_pred = np.array(y_pred)
    


    # Calculate original metrics
    r2_score(y_true, y_pred)

    # Binary classification metrics
    y_true_binary = (y_true >= 0.5).astype(int)
    y_pred_binary = (y_pred >= 0.5).astype(int)

    # Calculate original accuracy
    accuracy_score(y_true_binary, y_pred_binary)

    # Calculate original AUROC
    original_auroc = roc_auc_score(y_true_binary, y_pred)

    # Initialize lists to store bootstrap results
    bootstrap_r2 = []
    bootstrap_accuracy = []
    bootstrap_auroc = []

    # Bootstrap iterations
    for _ in range(n_iterations):
        # Generate bootstrap sample indices
        indices = resample(range(len(y_true)), replace=True, n_samples=len(y_true))

        # Get bootstrap samples
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        # Calculate R² on bootstrap sample
        boot_r2 = r2_score(y_true_boot, y_pred_boot)
        bootstrap_r2.append(boot_r2)

        # Calculate binary metrics on bootstrap sample
        y_true_binary_boot = (y_true_boot >= 0.5).astype(int)
        y_pred_binary_boot = (y_pred_boot >= 0.5).astype(int)

        # Calculate accuracy on bootstrap sample
        boot_accuracy = accuracy_score(y_true_binary_boot, y_pred_binary_boot)
        bootstrap_accuracy.append(boot_accuracy)

        # Calculate AUROC on bootstrap sample
        boot_auroc = roc_auc_score(y_true_binary_boot, y_pred_boot)
        bootstrap_auroc.append(boot_auroc)

    # Calculate 95% confidence intervals
    np.percentile(bootstrap_r2, [2.5, 97.5])
    np.percentile(bootstrap_accuracy, [2.5, 97.5])
    auroc_ci = np.percentile(bootstrap_auroc, [2.5, 97.5])

    # Return results
    return {
    "r2": r2_score(y_true, y_pred),
    "r2_ci": np.percentile(bootstrap_r2, [2.5, 97.5]),
    "accuracy": accuracy_score(y_true_binary, y_pred_binary),
    "accuracy_ci": np.percentile(bootstrap_accuracy, [2.5, 97.5]),
    "auroc": original_auroc,
    "auroc_ci": auroc_ci,
    }


def knn_replace_step(model, model_path):

    knn_data_name = "_8votes"
    recalc_knn = True

    if model_path != None:
        name = model_path.split("/")[-2] + "_" + model_path.split("/")[-1][:-4]

        if os.path.exists(f"./model_knn_layers/{name}{knn_data_name}.pth"):

            model_dict = torch.load(f"./model_knn_layers/{name}{knn_data_name}.pth")
            output_tensor = model_dict["prototype_tensor"]
            proto_labels = model_dict["proto_labels"]
            input_ids = model_dict["input_ids"]
            recalc_knn = False

        else:
            recalc_knn = True

    if recalc_knn:

        customDataSet_kw_args = {
            "eeg_data": {
                "train": "/home/muhammad-adeel-ajmal-khan/Documents/snd/real_npy",
                "train_push": "/home/muhammad-adeel-ajmal-khan/Documents/snd/real_npy",
                "eval": "/home/muhammad-adeel-ajmal-khan/Documents/snd/real_npy",
            },
            "labels": {
                "train": "/home/muhammad-adeel-ajmal-khan/Documents/repos/ProtoEEG/sn2_data/organized_data/sn2_train_labels.npy",
                "train_push": "/home/muhammad-adeel-ajmal-khan/Documents/repos/ProtoEEG/sn2_data/organized_data/sn2_train_labels.npy",
                "eval": "/home/muhammad-adeel-ajmal-khan/Documents/repos/ProtoEEG/sn2_data/organized_data/sn2_train_labels.npy",
            },
            "threshold": 0.5,
            "train_transform": None,
            "push_transform": "spikenet_helpers.eeg_crop spikenet_helpers.spikenet_transform",
            "eval_transform": "spikenet_helpers.eeg_crop spikenet_helpers.spikenet_transform spikenet_helpers.extremes_remover",
        }

        knn_dataset = EEG_ConcatDataset(mode="eval", **customDataSet_kw_args)

        knn_loader_config = {"batch_size": 128, "shuffle": False, "pin_memory": False}

        knn_loader = torch.utils.data.DataLoader(knn_dataset, **knn_loader_config)

        output_tensor = torch.tensor([])  # Initialize empty tensor
        input_ids = []  # Initialize empty list
        proto_labels = []

        with torch.no_grad():
            for sample in knn_loader:
                input = sample["img"]
                input_ids += sample["sample_id"]  # Add IDs to list
                proto_labels += sample["target"]

                x = model.backbone(input.cuda())
                x = model.add_on_layers(x).detach().cpu()

                output_tensor = torch.cat((output_tensor, x), dim=0)

        data_dict = {
            "prototype_tensor": output_tensor,
            "proto_labels": proto_labels,
            "input_ids": input_ids,
        }

        if model_path != None:
            torch.save(data_dict, f"./model_knn_layers/{name}{knn_data_name}.pth")

    # step 2 - set the prototype layer to exacty equal the backbone
    model.prototype_layer.prototype_tensors.data = output_tensor.cuda()

    proto_labels = [i.item() for i in proto_labels]
    return model, proto_labels, input_ids


def get_demo_data():
    
    customDataSet_kw_args = {
        "eeg_data": {
            "train": "sample_data/sample_data.pth",
            "train_push": "sample_data/sample_data.pth",
            "eval": "sample_data/sample_data.pth",
        },
        "labels": {
            "train": "sample_data/sample_labels.npy",
            "train_push": "sample_data/sample_labels.npy",
            "eval": "sample_data/sample_labels.npy",
        },
        "threshold": 0.5,
        "train_transform": None,
        "push_transform": "spikenet_helpers.eeg_crop spikenet_helpers.spikenet_transform",
        "eval_transform": f"spikenet_helpers.eeg_crop spikenet_helpers.spikenet_transform spikenet_helpers.extremes_remover",
    }

    test_dataset = EEG_ConcatDataset(mode="eval", **customDataSet_kw_args)

    test_loader_config = {"batch_size": 64, "shuffle": False, "pin_memory": False}

    return torch.utils.data.DataLoader(test_dataset, **test_loader_config)


def get_test_data(name):
    
    if name == "8votes":
        data_dict = "test"
        test_data_name = "_8votes_test"
    elif name == "ez":
        data_dict = "test"
        test_data_name = "_ez_test"
    elif name == "val":
        data_dict = "train"
        test_data_name = "_8votes_val"
    else:
        raise ValueError(f"Invalid name: '{name}'. Expected '8votes', 'ez', or 'val'.")

    customDataSet_kw_args = {
        "eeg_data": {
            "train": "/home/muhammad-adeel-ajmal-khan/Documents/snd/real_npy",
            "train_push": "/home/muhammad-adeel-ajmal-khan/Documents/snd/real_npy",
            "eval": "/home/muhammad-adeel-ajmal-khan/Documents/snd/real_npy",
        },
        "labels": {
            "train": "../sn2_data/organized_data/sn2_train_labels.npy",
            "train_push": "../sn2_data/organized_data/sn2_train_labels.npy",
            "eval": f"../sn2_data/organized_data/sn2{test_data_name}_labels.npy",
        },
        "threshold": 0.5,
        "train_transform": None,
        "push_transform": "spikenet_helpers.eeg_crop spikenet_helpers.spikenet_transform",
        "eval_transform": f"spikenet_helpers.eeg_crop spikenet_helpers.spikenet_transform spikenet_helpers.extremes_remover",
    }

    test_dataset = EEG_ConcatDataset(mode="eval", **customDataSet_kw_args)

    test_loader_config = {"batch_size": 64, "shuffle": False, "pin_memory": False}

    return torch.utils.data.DataLoader(test_dataset, **test_loader_config)
