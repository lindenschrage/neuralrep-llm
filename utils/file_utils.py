import json
import os
import pickle
import numpy as np

def load_json(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        return {}

def save_json(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def load_pickle(file_path):
    """Load a pickle file."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pickle(file_path, data):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def mean_excluding_diagonal(matrix):
    return np.mean([matrix[i][j] for i in range(len(matrix)) for j in range(len(matrix)) if i != j])
