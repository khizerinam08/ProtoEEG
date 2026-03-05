import pandas as pd

# 1. Load your existing Candidate file (The ~12k files)
df_candidates = pd.read_csv('labels_8votes.csv', sep=';')

# 2. Load your Control/Hard Negative file list
# Assuming you have a list or CSV of the 4,457 control segment filenames
df_controls = pd.read_csv('controlset.csv') 

# 3. Format the Controls to match Candidates
# Note: You likely need to generate the specific 'event_file' names if you haven't already.
# If df_controls represents full EEGs, you effectively take the hard negative segments extracted from them.
# If df_controls already lists the segments:
df_controls['fraction_of_yes'] = 0.0
df_controls['total_votes_received'] = 8  # Implies 0/8 votes were positive

# Ensure columns match exactly
df_controls = df_controls[['event_file', 'eeg_file', 'total_votes_received', 'fraction_of_yes', 'Mode']]

# 4. Combine them
final_dataset = pd.concat([df_candidates, df_controls], axis=0)

# 5. Verify the total
print(f"Total Files: {len(final_dataset)}")
# Should be ~17,368 (12,911 candidates + 4,457 controls)

final_dataset.to_csv('labels_8votes_with_controls.csv', index=False)
