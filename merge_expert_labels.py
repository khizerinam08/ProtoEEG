import pandas as pd

# 1. Load the labels_8votes.csv (the 12,911 samples with >= 8 votes)
try:
    df_8votes = pd.read_csv('labels_8votes.csv', sep=';')
    print(f"Loaded labels_8votes.csv: {len(df_8votes)} samples (>= 8 votes)")
except FileNotFoundError:
    print("Error: labels_8votes.csv not found. Please run filter_votes.py first.")
    exit(1)

# 2. Load the original lut file to extract the 2-expert/0-spike samples
df_lut = pd.read_csv('lut_labelled_20230628.csv', sep=';')

# 3. Filter for 2 experts and 0 spike votes (fraction_of_yes == 0.0)
df_expert_negatives = df_lut[
    (df_lut['total_votes_received'] == 2) & 
    (df_lut['fraction_of_yes'] == 0.0)
]

# Take only the first 5000 samples
df_expert_negatives_5k = df_expert_negatives.head(5536)

print(f"Found total 2-expert negative samples: {len(df_expert_negatives)}")
print(f"Taking the first {len(df_expert_negatives_5k)} samples.")

# 4. Merge them
merged_df = pd.concat([df_8votes, df_expert_negatives_5k], ignore_index=True)

# 5. Save the result
output_filename = 'labels_8votes_plus_5k_expert_negatives.csv'
merged_df.to_csv(output_filename, index=False, sep=';')

print(f"\n========================================")
print(f"SUCCESS!")
print(f"Total merged samples: {len(merged_df)}")
print(f"Result saved to: {output_filename}")
print(f"========================================")
