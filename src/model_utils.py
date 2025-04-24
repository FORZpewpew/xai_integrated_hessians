import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import matplotlib.pyplot as plt
import seaborn as sns
import spacy

nlp = spacy.load("en_core_web_sm")
negation_words = ["not", "no", "n't"]

def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def visualize_attributions(attributions, tokens, valid_length):
    """
    Visualizes the attributions as a bar chart.
    
    Args:
        attributions: The computed attributions (1, seq_len, embed_dim).
        tokens: The decoded tokens (list of strings).
        valid_length: The valid sequence length (excluding padding).
    """
    # Reduce the attribution vector by averaging across the embedding dimension (768)
    attributions = attributions.squeeze(0)  # shape: (seq_len, embed_dim)
    attributions = attributions[:valid_length]  # Ignore padding
    
    # Average across the embedding dimension and detach
    token_attributions = attributions.mean(dim=1).detach().cpu().numpy()  # shape: (seq_len,)

    # Get the valid tokens (non-padding)
    valid_tokens = tokens[:valid_length]

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.bar(valid_tokens, token_attributions, color='blue')
    plt.xlabel('Tokens')
    plt.ylabel('Attribution')
    plt.xticks(rotation=45, ha='right')
    plt.title("Token Importance based on Integrated Gradients")
    plt.show()


def visualize_hessians(hessians, tokens):
    print(tokens)

    hessians = hessians[:len(tokens), :len(tokens)]
    hessians = hessians.detach().cpu().numpy()

    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(hessians, xticklabels=tokens, yticklabels=tokens, cmap="coolwarm", annot=False)
    plt.title("Integrated Hessians Interaction Matrix")
    plt.xlabel("Tokens")
    plt.ylabel("Tokens")
    plt.show()


def visualize_filtered_hessians(hessians, tokens, valid_length):
    """
    Visualizes the Hessians heatmap for informative tokens only.
    
    Args:
        hessians: Hessians tensor of shape (seq_len, seq_len)
        tokens: List of decoded tokens
        valid_length: Number of valid tokens (non-padding)
    """
    # Trim tokens and Hessians to valid length
    tokens = tokens[:valid_length]
    hessians = hessians[:valid_length, :valid_length]

    # Get POS tags
    pos_tags = get_pos_tags(tokens)
    
    # Filter tokens and keep their indices
    filtered_tokens = []
    filtered_indices = []
    for i, (token, tag) in enumerate(zip(tokens, pos_tags)):
        if tag in ['PUNCT', 'DET', 'ADP', 'AUX'] and token.lower() not in negation_words:
            continue
        filtered_tokens.append(token)
        filtered_indices.append(i)

    if len(filtered_indices) < 2:
        print("Not enough informative tokens to plot.")
        return

    # Slice the Hessians matrix to only include filtered tokens
    filtered_hessians = hessians[filtered_indices][:, filtered_indices]
    filtered_hessians = filtered_hessians.detach().cpu().numpy()

    # Plot the filtered heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(filtered_hessians, xticklabels=filtered_tokens, yticklabels=filtered_tokens, cmap="coolwarm", annot=True)
    plt.title("Filtered Integrated Hessians Interaction Matrix")
    plt.xlabel("Tokens")
    plt.ylabel("Tokens")
    plt.tight_layout()
    plt.show()


def filter_tokens(tokens, pos_tags):
    """
    Filters out non-POS tokens (stopwords, punctuation) except for negations.
    """
    filtered_tokens = []
    for i, token in enumerate(tokens):
        # Exclude punctuation and stopwords unless it's a negation word
        if pos_tags[i] in ['PUNCT', 'DET', 'ADP', 'AUX'] and token.lower() not in negation_words:
            continue
        filtered_tokens.append(token)
    
    return filtered_tokens

def get_pos_tags(tokens):
    """
    Get the POS tags for the tokens using spaCy.
    """
    doc = nlp(" ".join(tokens))
    return [token.pos_ for token in doc]


def find_most_important_tokens_from_hessians(hessians, tokens, valid_length):
    tokens = tokens[:valid_length]
    hessians = hessians[:valid_length, :valid_length]

    pos_tags = get_pos_tags(tokens)

    filtered_tokens = []
    filtered_indices = []
    for i, (token, tag) in enumerate(zip(tokens, pos_tags)):
        if tag in ['PUNCT', 'DET', 'ADP', 'AUX'] and token.lower() not in negation_words:
            continue
        filtered_tokens.append(token)
        filtered_indices.append(i)

    if len(filtered_indices) < 2:
        print("Not enough informative tokens for interaction analysis.")
        return None

    # Slice the matrix once into filtered form (like in visualization)
    filtered_hessians = hessians[filtered_indices][:, filtered_indices]

    # Find max interaction in filtered matrix
    max_val = 0
    max_pair = (0, 1)
    for i in range(len(filtered_tokens)):
        for j in range(i + 1, len(filtered_tokens)):
            val = abs(filtered_hessians[i, j].item())
            if val > max_val:
                max_val = val
                max_pair = (i, j)

    print("Most important tokens (by interaction):", 
          filtered_tokens[max_pair[0]], 
          filtered_tokens[max_pair[1]])
    return filtered_tokens[max_pair[0]], filtered_tokens[max_pair[1]]
