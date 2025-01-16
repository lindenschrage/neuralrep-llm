import os
import argparse
import torch
import numpy as np
import pandas as pd


from config import *
from modules.generate_sentences import generate_sentences
from modules.generate_embeddings import generate_manifold_embeddings
from modules.principal_component import compute_principal_components
from modules.compute_results import (
    get_feature_list_for_layer,
    get_SNR_matrix_numpy,
    get_x_M,
)
from modules.compute_error import get_error_list_5shot_vectorized
from utils.file_utils import load_pickle
from modules.plot import (
    plot_snr_vs_error, 
    plot_average_metrics
)

##Set device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cpu')


def main():
    """Main function to execute the data processing and plotting."""
    ## Parse arguments for layer
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True, help="Layer number to process")
    args = parser.parse_args()
    selected_layer = args.layer

    ## Generate sentences
    print("Script starting...")
    generate_sentences()

    ## Load sentences
    manifold_sentences = load_pickle(MANIFOLD_SENTENCES_FILE)

    ## Generate embeddings
    generate_manifold_embeddings(
        manifold_sentences, LAYERS, LLAMA_MODEL_PATH, LLAMA_API_KEY
    )

    ## Load embeddings
    manifold_average_embedding_file = os.path.join(
        EMBEDDINGS_DIR, 'manifold_average_embeddings.pt'
    )
    manifold_average_embeddings = torch.load(manifold_average_embedding_file)

    ## Compute principal components
    compute_principal_components(manifold_average_embeddings, LAYERS)

    ## Process selected layer
    SNR, ERROR = process_layer(manifold_average_embeddings, selected_layer)

    ## Plot SNR vs ERROR
    plot_snr_vs_error(SNR, ERROR, selected_layer)

    ## Plot average metrics per layer, only run once data for ALL layers is computed
    #plot_average_metrics()


def process_layer(manifold_average_embeddings, selected_layer):
    """Compute metrics for the selected layer and save results."""
    # Compute features for the selected layer
    feature_list = get_feature_list_for_layer(
        manifold_average_embeddings, selected_layer, CATEGORIES
    )
    feature_list = np.asarray(feature_list)  # Shape: (num_categories, 200, 4096)

    # Compute SNR and related metrics
    SNR, signal, bias, vdim, vbias, signoise, nnoise, Da_list = get_SNR_matrix_numpy(
        torch.tensor(feature_list), device
    )
    SNR = get_x_M(SNR)

    results = {
        'layer': selected_layer,
        'snr': SNR,
        'signal': signal,
        'bias': bias,
        'overlap': signoise,
        'dimension': Da_list,
    }

    # Save results
    np.save(f'final/average_manifold/results_{selected_layer}.npy', results)

    mean_SNR = np.nanmean(SNR)
    print('MEAN SNR LAYER', selected_layer, mean_SNR)

    # Compute error
    ERROR = get_error_list_5shot_vectorized(torch.tensor(feature_list))
    np.save(f'final/average_manifold/error_{selected_layer}.npy', ERROR)

    mean_ERROR = np.nanmean(ERROR)
    print('MEAN ERROR LAYER', selected_layer, mean_ERROR)

    # Save metrics to CSV
    metrics_file = f'final/average_manifold/metrics_{selected_layer}.csv'
    with open(metrics_file, 'w') as f:
        f.write("layer,mean_snr,mean_error\n")
        f.write(f"{selected_layer},{mean_SNR},{mean_ERROR}\n")
    
    return SNR, ERROR



if __name__ == "__main__":
    main()