import os
import torch
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import EMBEDDINGS_DIR, TRANSFORMERS_CACHE

def generate_manifold_embeddings(manifold_sentences, LAYERS, LLAMA_VERSION, access_token):
    llama_tokenizer = AutoTokenizer.from_pretrained(
        LLAMA_VERSION, 
        use_auth_token=access_token,
        cache_dir=TRANSFORMERS_CACHE)
    llama_model = AutoModelForCausalLM.from_pretrained(
        LLAMA_VERSION,
        use_auth_token=access_token,
        output_hidden_states=True,
        cache_dir=TRANSFORMERS_CACHE).to('cuda')

    average_embedding_file = os.path.join(EMBEDDINGS_DIR, 'manifold_average_embeddings.pt')
    average_embedding_dict = {layer: {} for layer in LAYERS}

    for key in manifold_sentences:
        print(f"Processing key: {key}")
        all_embeddings = {layer: [] for layer in LAYERS}
        for sentence in manifold_sentences[key]:
            tokens = llama_tokenizer.encode(sentence, return_tensors="pt").to('cuda')
            with torch.no_grad():
                outputs = llama_model(tokens, output_hidden_states=True, return_dict=True)
                hidden_states = outputs.hidden_states

            for layer in LAYERS:
                avg_token_embedding = torch.mean(hidden_states[layer][0, :, :].detach(), dim=0)
                all_embeddings[layer].append(avg_token_embedding.cpu().numpy())
        for layer in LAYERS:
            average_embedding_dict[layer][key] = all_embeddings[layer]

        torch.save(average_embedding_dict, average_embedding_file)