# neuralrep-llm
Official repository for the paper "Neural Representational Geometry of Concepts in Large Language Models" (Neurips 2024 Workshop on Symmetry and Geometry in Neural Representations)

For more information about the workshop, please visit the [NeurReps website](https://www.neurreps.org/about).

Below is a high-level overview of each relevant file. The code is structured into modules based on functionality:

- <strong>main.py</strong>:  
  The primary entry point that orchestrates:
  1. Generating sentences (if not already generated).  
  2. Generating embeddings from a LLaMA model.  
  3. Computing principal components.  
  4. Computing SNR and Error metrics.  
  5. Saving and plotting the results.

- <strong>modules/generate_sentences.py</strong>:  
  Uses the OpenAI GPT-based model to generate category-specific sentences (manifold sentences) for training and evaluation.  

  #### Category Definitions
  The code assumes that the sentences have already been generated using GPT-4 using the prompt found in `generate_categories.py`.  
  The category types and amount are specified in `config.py`.

- <strong>modules/generate_embeddings.py</strong>:  
  Generates embeddings using a LLaMA model via huggingface transformers. Each sentence is encoded, and the averaging of token embeddings is done per layer.

- <strong>modules/principal_component.py</strong>:  
  Removes the first principal component from the embeddings to reduce bias (if needed).  

- <strong>modules/compute_results.py</strong>:  
  Provides functions to compute:
  1. Feature arrays for categories at a specified layer.  
  2. SNR matrix using the m-shot approach.

- <strong>modules/compute_error.py</strong>:  
  Implements the 5-shot error computations using vectorized approaches in PyTorch.

- <strong>modules/plot.py</strong>:  
  Contains plotting logic for SNR vs. Error as well as average metrics across layers.

- <strong>utils/file_utils.py</strong>:  
  Utility methods for reading and writing JSON, Pickle files, etc.

- <strong>config.py</strong>:  
  Centralizes all directory paths, environment variables, constants (like layer indices, categories, number of sentences, etc). 

---

## Requirements

Below are some of the main packages and their versions taken from our conda environment file (environment.yml):

- Python: 3.10.14
- PyTorch: 2.2.2
- Transformers: 4.47.0
- Accelerate: 0.29.3
- Datasets: 2.19.0
- HuggingFace Hub: 0.26.5
- PyYAML: 6.0.1
- scikit-learn: 1.4.2
- TensorFlow: 2.16.1
- wandb: 0.16.6

These and other dependencies can be found in the [environment.yml](environment.yml) file. To create and activate this environment:

```bash
conda env create -f environment.yml
conda activate neurrep-llm-env
```

This setup ensures that all dependencies are installed and compatible with the codebase.

---

## How to Run

1. Ensure all environment variables are set (e.g., in .env or via export commands for your shell).  
2. From the project root, run:
   ```bash
   python main.py --layer 0
   ```  
   Replace "0" with the layer of interest for which you want to compute SNR and error metrics. For best results, you may want to run all layers:
   ```bash
   for L in 0 5 10 15 20 25 32; do
       python main.py --layer $L
   done
   ```
3. If needed, you can generate sentences separately first:
   ```bash
   python modules/generate_sentences.py
   ```  
   This step will create new .pkl files with the sentences if they do not already exist.

---

## Plotting and Results

• After running "main.py --layer {layer}".  
• Two main types of plots are generated:  
  1. snr_error_{layer}.png  
  2. average_error_per_layer.png and average_snr_per_layer.png (generated once data for all layers is available).  

• Numerical outputs include:  
  - results_{layer}.npy (SNR, signal, bias, dimension)  
  - error_{layer}.npy (matrix of 5-shot errors)  
  - metrics_{layer}.csv (mean SNR and error summaries)  

---

