import torch
from torch.autograd import grad
from tqdm import tqdm
from torch.nn.attention import sdpa_kernel, SDPBackend

# Set a fixed random seed for reproducibility
torch.manual_seed(42)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

def integrated_hessians(model, inputs, baseline, steps=50, target_class=1):
    """
    Compute Integrated Hessians for a single input.
    Args:
        model: PyTorch model
        inputs: input tensor (1, seq_len)
        baseline: baseline tensor (1, seq_len)
        steps: number of Riemann steps
        target_class: class index to attribute
    Returns:
        ih: (seq_len, seq_len) interaction matrix
    """
    # Dynamically access the embedding layer based on the model type
    if hasattr(model, 'bert'):
        embeddings = model.bert.embeddings.word_embeddings
    elif hasattr(model, 'roberta'):
        embeddings = model.roberta.embeddings.word_embeddings
    elif hasattr(model, 'distilbert'):
        embeddings = model.distilbert.embeddings.word_embeddings
    else:
        raise AttributeError(f"The model does not have a recognized embedding layer: {model}.")

    device = next(model.parameters()).device

    with sdpa_kernel([SDPBackend.MATH]):
        # Compute token embeddings for input and baseline
        input_embed = embeddings(inputs).detach()
        baseline_embed = embeddings(baseline).detach()

        # print(f"Input embedding shape: {input_embed.shape}")
        # print(f"Baseline embedding shape: {baseline_embed.shape}")

        seq_len, embed_dim = input_embed.shape[1], input_embed.shape[2]
        # print(f"Sequence length: {seq_len}, Embedding dimension: {embed_dim}")

        ih = torch.zeros((seq_len, seq_len), device=device)

        # Use tqdm for progress tracking
        for alpha in tqdm(torch.linspace(0, 1, steps, device=device), desc="Processing alphas", unit="step"):
            # Interpolate embeddings
            emb_alpha = baseline_embed + alpha * (input_embed - baseline_embed)
            emb_alpha.requires_grad_(True)

            # Compute predictions
            logits = model(inputs_embeds=emb_alpha).logits
            # print(f"Logits: {logits}")
            score = logits[0, target_class]

            # First gradients w.r.t. embeddings
            grads = grad(score, emb_alpha, create_graph=True)[0]  # (1, seq_len, embed_dim)
            # print(f"Gradients shape: {grads.shape}")

            # Second derivative: Hessian-vector products
            for i in range(seq_len):
                # print(f"Processing token {i}/{seq_len}")
                # Recompute emb_alpha to avoid retain_graph=True
                emb_alpha = baseline_embed + alpha * (input_embed - baseline_embed)
                emb_alpha.requires_grad_(True)

                # Recompute first gradients
                logits = model(inputs_embeds=emb_alpha).logits
                score = logits[0, target_class]
                grads = grad(score, emb_alpha, create_graph=True)[0]

                # gradient of grad_i w.r.t. emb_alpha
                grad_i = grads[0, i].unsqueeze(0)  # (1, embed_dim)
                # print(f"grad_i shape: {grad_i.shape}")
                hess = grad(grad_i, emb_alpha, grad_outputs=torch.ones_like(grad_i))[0]
                # print(f"Hessian shape: {hess.shape}")
                # Accumulate interaction: multiply by (x - x0)
                delta = (input_embed - baseline_embed)[0]
                # Adjust dimensions for compatibility and remove extra dimension
                ih[i] += (hess * delta).sum(dim=2).squeeze(0)

    # Scale by step size
    ih = ih * (1.0 / steps)
    return ih.cpu().detach()



def integrated_hessians_full(model, inputs, baseline, steps=50, target_class=1):
    """
    Compute the full Integrated Hessians for a single input using double integration.
    
    Args:
        model: PyTorch model
        inputs: input tensor of token indices, shape (1, seq_len)
        baseline: baseline tensor of token indices, shape (1, seq_len)
        steps: number of integration steps (both alpha and beta)
        target_class: class index for attribution
        
    Returns:
        ih: Tensor of shape (seq_len, seq_len), full integrated Hessian matrix
    """
    if hasattr(model, 'bert'):
        embeddings = model.bert.embeddings.word_embeddings
    elif hasattr(model, 'roberta'):
        embeddings = model.roberta.embeddings.word_embeddings
    elif hasattr(model, 'distilbert'):
        embeddings = model.distilbert.embeddings.word_embeddings
    else:
        raise AttributeError(f"The model does not have a recognized embedding layer: {model}.")


    device = next(model.parameters()).device
    input_embed = embeddings(inputs).detach()
    baseline_embed = embeddings(baseline).detach()
    delta = (input_embed - baseline_embed)[0]  # (seq_len, embed_dim)

    seq_len = input_embed.shape[1]
    ih = torch.zeros((seq_len, seq_len), device=device)

    alpha_vals = torch.linspace(0, 1, steps, device=device)
    beta_vals = torch.linspace(0, 1, steps, device=device)

    for alpha in tqdm(alpha_vals, desc="Outer alpha loop"):
        for beta in beta_vals:
            gamma = alpha * beta
            emb_gamma = baseline_embed + gamma * (input_embed - baseline_embed)
            emb_gamma.requires_grad_(True)

            logits = model(inputs_embeds=emb_gamma).logits
            score = logits[0, target_class]

            grads = grad(score, emb_gamma, create_graph=True, retain_graph=True)[0]  # (1, seq_len, embed_dim)

            for i in range(seq_len):
                grad_i = grads[0, i].unsqueeze(0)  # (1, embed_dim)
                hess_row = grad(grad_i, emb_gamma, grad_outputs=torch.ones_like(grad_i), retain_graph=True)[0][0]  # (seq_len, embed_dim)

                # Compute contribution to each (i,j) using delta_i * delta_j * Hess_ij
                for j in range(seq_len):
                    dot = torch.dot(hess_row[j], delta[j])  # scalar
                    ih[i, j] += delta[i].dot(delta[j]) * dot

    ih = ih / (steps * steps)  # Double integral average
    return ih.cpu().detach()
