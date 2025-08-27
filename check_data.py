import torch
import glob
import os

files = sorted(glob.glob('data/vecs/*.pt'))
valid_files = []

for f in files:
    try:
        data = torch.load(f, map_location='cpu')
        if isinstance(data, dict) and 'input_ids' in data and 'hidden_states' in data:
            valid_files.append(f)
            print(f'Valid: {os.path.basename(f)} - batch_size: {data["input_ids"].shape[0]}, seq_len: {data["input_ids"].shape[1]}')
    except Exception as e:
        print(f'Invalid: {os.path.basename(f)} - {e}')

print(f'\nTotal valid files: {len(valid_files)}')
if len(valid_files) > 0:
    print(f'First valid file: {valid_files[0]}')