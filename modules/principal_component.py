# modules/manifold_processor.py

import os
import torch
import numpy as np
from config import EMBEDDINGS_DIR


def get_covariance_matrix_total(embedding_dict):
    embeddings_list = []
    for key in embedding_dict:
        embeddings = embedding_dict[key]
        
        if isinstance(embeddings[0], torch.Tensor):
            embeddings = [embedding.cpu().numpy() for embedding in embeddings]
        
        embeddings = np.array(embeddings) 
        embeddings_list.append(embeddings)
    all_embeddings_matrix = np.vstack(embeddings_list)  # Shape: (4250, 4096)
    global_mean = np.mean(all_embeddings_matrix, axis=0)  # Shape: (4096,)
    print('mean shape', np.shape(global_mean))
    centered_embeddings_matrix = all_embeddings_matrix - global_mean
    covariance_matrix = (centered_embeddings_matrix.T @ centered_embeddings_matrix) / (centered_embeddings_matrix.shape[0] - 1)
    print('covariance_matrix shape', np.shape(covariance_matrix))
    return covariance_matrix

def total_PCA(covariance_matrix):
    eigenvals, eigenvecs = np.linalg.eigh(covariance_matrix)
    sorted_indices = np.argsort(eigenvals)[::-1]
    sorted_eigenvals = eigenvals[sorted_indices]
    print('sorted eigenvals', sorted_eigenvals[:10])
    sorted_eigenvecs = eigenvecs[:, sorted_indices]
    principal_eigenvec = sorted_eigenvecs[:, 0]
    print('PC norm', np.linalg.norm(principal_eigenvec))
    return principal_eigenvec

def updated_embeddings(embedding_dict):
    covariance_matrix = get_covariance_matrix_total(embedding_dict)
    principal_component = total_PCA(covariance_matrix)
    updated_embedding_dict = {}
    for key in embedding_dict:
        embeddings = embedding_dict[key]
        
        if isinstance(embeddings[0], torch.Tensor):
            embeddings = [embedding.cpu().numpy() for embedding in embeddings]
        
        embeddings = np.array(embeddings) 
        updated_embedding_list = []
        for embedding in embeddings:
            projection = np.dot(embedding, principal_component) * principal_component
            updated_embedding = embedding - projection
            updated_embedding_list.append(updated_embedding)
        updated_embedding_dict[key] = updated_embedding_list
    return principal_component, updated_embedding_dict


def compute_principal_components(embedding_dict, LAYERS):
    new_embedding_dict = {}
    principal_component_dict = {}

    for layer in LAYERS:
        print('Processing layer:', layer)
        principal_component, updated_embedding_dict = updated_embeddings(embedding_dict[layer])
        new_embedding_dict[layer] = updated_embedding_dict
        principal_component_dict[layer] = principal_component

    principal_components_file = os.path.join(EMBEDDINGS_DIR, 'principal_components.pt')
    manifold_update_embeddings_file = os.path.join(EMBEDDINGS_DIR, 'manifold_updated_embeddings.pt')
    
    torch.save(principal_component_dict, principal_components_file)
    torch.save(new_embedding_dict, manifold_update_embeddings_file)


def update_test_embeddings(test_embedding_dict, pc_dict):
    for layer, embeddings in test_embedding_dict.items():
        for item in embeddings:
            embedding = item['embedding']
            projection = np.dot(embedding, pc_dict[layer]) * pc_dict[layer]
            updated_embedding = embedding - projection
            item['embedding'] = updated_embedding
    test_updated_embeddings_file = os.path.join(EMBEDDINGS_DIR, 'test_updated_embeddings.pt')
    torch.save(test_embedding_dict, test_updated_embeddings_file)
