import pandas as pd
import os

# --- CONFIGURATION ---
INPUT_CSV = 'lut_labelled_20230628.csv'        # Your original CSV file
OUTPUT_CSV = 'labels_8votes.csv' # The new filtered file
REQUIRED_VOTES = 8

# 1. Read the file (assuming semicolon delimiter based on your previous messages)
print(f"Reading {INPUT_CSV}...")
try:
    df = pd.read_csv(INPUT_CSV, sep=';')
except FileNotFoundError:
    print("Error: Could not find labels.csv. Make sure it's in this folder.")
    exit()

# 2. Check the columns
if 'total_votes_received' not in df.columns:
    print("Error: Column 'total_votes_received' not found. Columns are:", df.columns)
    exit()

# 3. Filter
original_count = len(df)
df_filtered = df[df['total_votes_received'] == REQUIRED_VOTES]
new_count = len(df_filtered)

# 4. Save
if new_count > 0:
    df_filtered.to_csv(OUTPUT_CSV, sep=';', index=False)
    print("="*40)
    print(f"SUCCESS!")
    print(f"Original samples: {original_count}")
    print(f"Samples with exactly {REQUIRED_VOTES} votes: {new_count}")
    print(f"Filtered CSV saved to: {OUTPUT_CSV}")
    print("="*40)
    print("Next Step: Use 'labels_8votes.csv' in your pack_data.py script.")
else:
    print("="*40)
    print("WARNING: No samples found with exactly 8 votes!")
    print("You may need to skip this filtering step and use all data instead.")
    print("="*40)