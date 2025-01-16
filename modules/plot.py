import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy
from scipy.special import erfc

from config import LAYERS

# Gaussian error function
def H(x):
    return erfc(x/np.sqrt(2))/2

def plot_snr_vs_error(SNR, ERROR, selected_layer):
    """Plot SNR versus Generalization Error for the selected layer."""
    plt.figure(figsize=(3, 3))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.scatter(SNR, ERROR, s=5, label='Empirical')

    # Define theoretical function H(x)
    tt = np.linspace(0.5, 8, 100)
    plt.plot(tt, H(tt), color='red', label='Theory') 

    plt.xlabel('SNR', fontsize=10)
    plt.ylabel('Generalization Error', fontsize=10)
    plt.title(f'Layer {selected_layer}', fontsize=12)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f'final/average_manifold/snr_error_{selected_layer}.png', dpi=300)
    plt.close()


def plot_average_metrics():
    """Plot average error and SNR across all layers."""
    layer_values = []
    mean_errors = []
    mean_snrs = []

    for selected_layer in LAYERS:
        metrics_file = f'final/average_manifold/metrics_{selected_layer}.csv'
        if os.path.exists(metrics_file):
            df = pd.read_csv(metrics_file)
            layer_values.append(df['layer'].iloc[0])
            mean_errors.append(df['mean_error'].iloc[0])
            mean_snrs.append(df['mean_snr'].iloc[0])
        else:
            print(f'Metrics file for layer {selected_layer} not found.')

    # Plot average error per layer
    plt.figure(figsize=(3, 3))
    plt.plot(layer_values, mean_errors, marker='o', color='#1a80bb')
    plt.title('Average Error per Layer')
    plt.xlabel('Layer')
    plt.ylabel('Average Error')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('final/average_manifold/average_error_per_layer.png', dpi=300)
    plt.close()

    # Plot average SNR per layer
    plt.figure(figsize=(3, 3))
    plt.plot(layer_values, mean_snrs, marker='o', color='#1a80bb')
    plt.title('Average SNR per Layer')
    plt.xlabel('Layer')
    plt.ylabel('Average SNR')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('final/average_manifold/average_snr_per_layer.png', dpi=300)
    plt.close()


