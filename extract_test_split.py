import pandas as pd

# Path to the source file
INPUT_CSV = 'lut_labelled_20230628.csv'
OUTPUT_CSV = 'lut_test_split_2_8_votes.csv'

print(f"Reading {INPUT_CSV}...")

try:
    # Read the original file
    df = pd.read_csv(INPUT_CSV, sep=';')
    
    # Filter for:
    # 1. Test mode
    # 2. Total votes in [2, 8]
    df_filtered = df[
        (df['Mode'] == 'Test') & 
        (df['total_votes_received'].isin([2, 8]))
    ]
    
    # Save to new CSV
    df_filtered.to_csv(OUTPUT_CSV, sep=';', index=False)
    
    # Break down counts
    count_2 = len(df_filtered[df_filtered['total_votes_received'] == 2])
    count_8 = len(df_filtered[df_filtered['total_votes_received'] == 8])
    
    print("\n========================================")
    print("SUCCESS!")
    print(f"Total samples in filtered Test split: {len(df_filtered)}")
    print(f" - 2-vote samples: {count_2}")
    print(f" - 8-vote samples: {count_8}")
    print(f"Saved to: {OUTPUT_CSV}")
    print("========================================")

except Exception as e:
    print(f"An error occurred: {e}")
