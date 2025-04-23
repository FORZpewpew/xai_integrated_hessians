import torch
from datasets import load_dataset
from integrated_gradients.integrated_hessians import integrated_hessians
from integrated_gradients.model_utils import load_model_and_tokenizer
import matplotlib.pyplot as plt
import numpy as np

def example_usage():
    """
    Example usage of Integrated Hessians library.
    """
    # Load BERT sentiment model and tokenizer
    model_name = 'bert-base-uncased'
    tokenizer, model = load_model_and_tokenizer(model_name)

    # Set device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load example text
    text = "I love this movie! It was fantastic and thrilling."

    # Tokenize input and baseline (all zeros => [PAD] token)
    tokens = tokenizer(text, return_tensors='pt', padding='max_length', max_length=128, truncation=True).to(device)
    input_ids = tokens['input_ids']
    pad_id = tokenizer.pad_token_id
    baseline_ids = torch.full_like(input_ids, pad_id).to(device)

    ih_matrix = integrated_hessians(model, input_ids, baseline_ids, steps=20, target_class=1)
    print('Integrated Hessians shape:', ih_matrix.shape)
    print(ih_matrix)

    # Decode tokens for visualization
    decoded_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Visualize and explain
    visualize_and_explain(ih_matrix, decoded_tokens, target_class=1)

def visualize_and_explain(ih_matrix, tokens, target_class):
    """
    Visualize the interaction matrix and provide textual explanation.
    Args:
        ih_matrix: Interaction matrix (seq_len, seq_len)
        tokens: List of tokens corresponding to the input
        target_class: Target class index
    """
    # Convert interaction matrix to numpy
    ih_matrix = ih_matrix.numpy()

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(ih_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Interaction Strength')
    plt.xticks(ticks=np.arange(len(tokens)), labels=tokens, rotation=90)
    plt.yticks(ticks=np.arange(len(tokens)), labels=tokens)
    plt.title(f'Integrated Hessians Interaction Matrix (Class {target_class})')
    plt.xlabel('Tokens')
    plt.ylabel('Tokens')
    plt.tight_layout()
    plt.show()

    # Generate textual explanation
    max_interaction = np.unravel_index(np.argmax(ih_matrix, axis=None), ih_matrix.shape)
    token1, token2 = tokens[max_interaction[0]], tokens[max_interaction[1]]
    print(f"The strongest interaction is between '{token1}' and '{token2}', contributing to class {target_class}.")

if __name__ == '__main__':
    example_usage()