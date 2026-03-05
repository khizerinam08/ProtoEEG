import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_FOLDER = '/home/muhammad-adeel-ajmal-khan/Documents/snd/real_npy'             # Folder containing your .npy files
LABEL_FILE = './lut_test_split_2_8_votes.csv'        # Your CSV file
OUTPUT_DIR = './sn2_data/organized_data/' # Where to save the packed files

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load your CSV with the correct delimiter
df = pd.read_csv(LABEL_FILE, sep=';')

label_list = []

print(f"Processing {len(df)} samples...")

# Track missing files to report at the end
missing_files = []

for index, row in tqdm(df.iterrows(), total=len(df)):
    # --- A. HANDLE FILE ID ---
    # Using 'event_file' as the unique ID (e.g., Bonobo00001_0_520)
    file_id = str(row['event_file']).strip()
    
    # Construct path: real/Bonobo00001_0_520.npy
    file_path = os.path.join(DATA_FOLDER, f"{file_id}.npy")
    
    if not os.path.exists(file_path):
        missing_files.append(file_path)
        continue

    # --- B. VERIFY EEG DATA EXISTS ---
    try:
        # Quick shape check without keeping data in memory
        eeg_array = np.load(file_path)
        
        if eeg_array.shape[1] < 160:
            print(f"Skipping {file_id}: Time dimension {eeg_array.shape[1]} too short.")
            continue
        
        del eeg_array  # Don't keep in memory
            
    except Exception as e:
        print(f"Error loading {file_id}: {e}")
        continue

    # --- C. RECONSTRUCT VOTES ---
    # The repo calculates: label = mean(votes) > threshold
    # You have 'total_votes_received' and 'fraction_of_yes'.
    # We must reconstruct the list of [1, 0, 1...] that produces that fraction.
    
    total_votes = int(row['total_votes_received'])
    fraction = float(row['fraction_of_yes'])
    
    # Calculate exact number of 'yes' votes
    yes_votes_count = int(round(fraction * total_votes))
    no_votes_count = total_votes - yes_votes_count
    
    # Create the binary list (e.g., [1, 1, 0, 0...])
    votes_array = np.array([1] * yes_votes_count + [0] * no_votes_count)
    
    # --- D. FORMAT LABELS FOR PROTOEEG ---
    # The Dataset class reads: self.labels[index][0][0] for ID
    # and self.labels[index][4][0] for votes.
    
    label_row = [
        [file_id],         # Index 0: ID wrapped in list
        row['eeg_file'],   # Index 1: Metadata (e.g., source file) - Unused by model but good to keep
        row['Mode'],       # Index 2: Metadata (e.g., Train) - Unused by model
        "placeholder",     # Index 3: Placeholder
        [votes_array]      # Index 4: Votes array wrapped in list
    ]
    
    # Append as an object array to preserve structure
    label_list.append(np.array(label_row, dtype=object))

# --- E. SAVE LABELS FILE ---
print(f"Successfully processed {len(label_list)} files.")
if missing_files:
    print(f"Warning: {len(missing_files)} files were defined in CSV but not found in '{DATA_FOLDER}'.")

# Save the Labels Array
final_labels = np.array(label_list, dtype=object)
np.save(os.path.join(OUTPUT_DIR, 'sn2_train_labels.npy'), final_labels)

print(f"Labels saved to: {OUTPUT_DIR}/sn2_train_labels.npy")
print(f"EEG data stays as raw .npy files in: {DATA_FOLDER}")