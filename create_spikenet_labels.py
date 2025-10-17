import torch
import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm

def leave_one_channel_in(signal):

    # signal: 1sec data chunk

    eeg = signal[0:19]

    # average

    x1 = eeg - eeg.mean(axis=0)

    # bipolar

    channels = [
        "Fp1",
        "F3",
        "C3",
        "P3",
        "F7",
        "T3",
        "T5",
        "O1",
        "Fz",
        "Cz",
        "Pz",
        "Fp2",
        "F4",
        "C4",
        "P4",
        "F8",
        "T4",
        "T6",
        "O2",
    ]

    bipolar_channels = [
        "Fp1-F7",
        "F7-T3",
        "T3-T5",
        "T5-O1",
        "Fp2-F8",
        "F8-T4",
        "T4-T6",
        "T6-O2",
        "Fp1-F3",
        "F3-C3",
        "C3-P3",
        "P3-O1",
        "Fp2-F4",
        "F4-C4",
        "C4-P4",
        "P4-O2",
        "Fz-Cz",
        "Cz-Pz",
    ]

    idx = np.array(
        [
            [channels.index(bc.split("-")[0]), channels.index(bc.split("-")[1])]
            for bc in bipolar_channels
        ]
    )

    x2 = eeg[idx[:, 0]] - eeg[idx[:, 1]]

    # concatenate and flatten

    x = np.array([*x1, *x2])

    z1 = x.ravel()

    # zero-pad the rest

    z2 = np.zeros((36, len(z1)))

    signal = np.array([z1, *z2])

    return signal


# load in the model weights (must use python 3.7 and keras 2.2.2)
with open("protopnet/pretrained/model_fold_1_structure.txt", "r") as fff:
    json_string = fff.read()
model = tf.keras.models.model_from_json(json_string)
model.load_weights("protopnet/pretrained/model_fold_1_weight.h5")


train_dict = torch.load("../sn2_data/organized_data/train_dict.pth")  # our data
test_dict = torch.load("../sn2_data/organized_data/test_dict.pth")
eeg_ids = list(train_dict.keys()) + list(test_dict.keys())


vals_dict = {}
count = 0
for eeg_id in tqdm(eeg_ids, desc="Processing EEG IDs"):

    # Initial preprocessing steps remain the same
    try:
        eeg = train_dict[eeg_id]  # [20, 192]
    except:
        eeg = test_dict[eeg_id]
    eeg = eeg[:, 32:160]  # grab center 128
    eeg = leave_one_channel_in(eeg)  # [37, 4736]



    # Convert to tensor and reshape
    eeg = torch.from_numpy(eeg).unsqueeze(0)

    # Reshape to [1, 37, 37, 128] using unfold
    batched_input = eeg.unfold(dimension=-1, size=128, step=128)
    batched_input = batched_input.squeeze(0).transpose(0, 1)

    ######!!! first 37 indexs the channels (1 real, 36 flat), 2nd 37 indexes the 37 of 1 eeg ###

    for i in range(37):
        # assert torch.abs(batched_input[i, 0, :]).mean() != 0

        for j in range(37):
            if j != 0:
                assert batched_input[i, j, :].mean() == 0

    # transpose is [37, 128, 37], unsqueezed(2) should be [37, 128, 1, 37]
    batched_input = batched_input.transpose(-1, -2).unsqueeze(2)

    input = tf.convert_to_tensor(batched_input)
    output = model.predict(input)[:, 0]

    out_as_tensor = torch.from_numpy(output).clone()

    vals_dict[eeg_id] = out_as_tensor

    del input
    del batched_input


torch.save(vals_dict, "model_feats/test.pth")
