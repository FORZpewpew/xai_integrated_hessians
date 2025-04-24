import torch

# Define Integrated Gradients function
def integrated_gradients(model, input_ids, baseline_ids, steps=50, target_class=None):
    """
    Compute Integrated Gradients for a given model and input.
    Args:
        model: The model to explain.
        input_ids: The input tensor (1, seq_len).
        baseline_ids: The baseline tensor (1, seq_len).
        steps: Number of steps for the Riemann sum approximation.
        target_class: The target class index (optional).
    Returns:
        A tensor of attributions (1, seq_len).
    """
    input_ids = input_ids.long()
    baseline_ids = baseline_ids.long()

    # Dynamically access the embedding layer based on the model type
    if hasattr(model, 'bert'):
        embeddings = model.bert.embeddings.word_embeddings
    elif hasattr(model, 'roberta'):
        embeddings = model.roberta.embeddings.word_embeddings
    elif hasattr(model, 'distilbert'):
        embeddings = model.distilbert.embeddings.word_embeddings
    else:
        raise AttributeError(f"Model does not have recognized embedding layer")

    # Convert input_ids and baseline_ids to embeddings
    input_embeddings = embeddings(input_ids)
    baseline_embeddings = embeddings(baseline_ids)

    # Generate scaled embeddings
    scaled_inputs = [baseline_embeddings + (float(i) / steps) * (input_embeddings - baseline_embeddings) for i in range(steps + 1)]
    scaled_inputs = torch.cat(scaled_inputs, dim=0)

    # Convert scaled_inputs into a leaf tensor with requires_grad
    scaled_inputs = torch.autograd.Variable(scaled_inputs, requires_grad=True)

    # Compute model outputs for each scaled input
    outputs = model(inputs_embeds=scaled_inputs)[0]  # shape: (steps+1, batch_size, num_classes)

    # If target_class is provided, select its score for attribution
    if target_class is not None:
        outputs = outputs[:, target_class]  # Select target_class column from outputs (shape: steps+1)

    # Compute gradients with respect to scaled inputs
    gradients = torch.autograd.grad(outputs.sum(), scaled_inputs)[0]  # gradients shape: (steps+1, 1, seq_len, embed_dim)

    # Average gradients across the steps (excluding the last step)
    avg_gradients = gradients[:-1].mean(dim=0)  # Shape: (1, seq_len, embed_dim)

    # Compute attributions
    attributions = (input_embeddings - baseline_embeddings) * avg_gradients
    return attributions
