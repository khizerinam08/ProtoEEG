from sklearn.metrics import roc_auc_score
from protopnet.knn_models import ProtoEEGkNN
from protopnet.eval_utils import get_test_data, bootstrap_metrics_ci, get_demo_data
import torch
import argparse
import numpy as np
import time

parser = argparse.ArgumentParser()
parser.add_argument("-path", help="path name in ./live/artifacts/")
parser.add_argument("-topk", type=int, default=10, help="topk value to use")
args = parser.parse_args()


model = ProtoEEGkNN(args.path, topk=args.topk)


# model.prototype_layer.importance_by_statistic.data = nn.Parameter(torch.log(torch.tensor([0.000001, 0.18, 0.18, 0.64], dtype=torch.float32))).cuda()
sm = torch.nn.Softmax(dim=0)
importance_stats = sm(model.base_model.prototype_layer.importance_by_statistic)
print("Model importance stats (latent, range, var, FFT): ", importance_stats)

test_loader = get_test_data("8votes")
#test_loader = get_demo_data()

y_true = np.array([])
y_pred = np.array([])
sample_names = []
nbrs = []
sample_dict = {}
for sample in test_loader:

    with torch.no_grad():

        # Spikenet takes in transposed inputs of 128, 37
        eeg = sample["img"]
        input_ids = sample["sample_id"]
        sample_names += input_ids
        target = sample["target"]
        y_true = np.concatenate((y_true, target))

        output_dict = model.forward(eeg, input_ids)

        prediction = output_dict["prediction"]
        y_pred = np.concatenate((y_pred, prediction))

        # nbrs += output_dict['neighbor_labels']
        # print(torch.abs(eeg).mean())

        for i in range(len(input_ids)):
            sample_dict[input_ids[i]] = {
                "label": target[i].item(),
                "prediction": prediction[i],
            }

results = bootstrap_metrics_ci(y_true, y_pred)

print("=" * 50)
print("Bootstrap Metrics with 95% Confidence Intervals")
print("=" * 50)
print(f"R² Score:    {results['r2']:.4f} (CI: [{results['r2_ci'][0]:.4f}, {results['r2_ci'][1]:.4f}])")
print(f"Accuracy:    {results['accuracy']:.4f} (CI: [{results['accuracy_ci'][0]:.4f}, {results['accuracy_ci'][1]:.4f}])")
print(f"AUROC:       {results['auroc']:.4f} (CI: [{results['auroc_ci'][0]:.4f}, {results['auroc_ci'][1]:.4f}])")
print("=" * 50)
