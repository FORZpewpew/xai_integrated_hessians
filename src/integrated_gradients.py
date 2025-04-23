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

    # Convert input_ids and baseline_ids to embeddings
    embeddings = model.bert.embeddings.word_embeddings
    input_embeddings = embeddings(input_ids)
    baseline_embeddings = embeddings(baseline_ids)

    # Generate scaled embeddings
    scaled_inputs = [baseline_embeddings + (float(i) / steps) * (input_embeddings - baseline_embeddings) for i in range(steps + 1)]
    scaled_inputs = torch.cat(scaled_inputs, dim=0)
    
    print(scaled_inputs)

    # Detach scaled_inputs to make it a leaf variable
    scaled_inputs = scaled_inputs.detach()
    scaled_inputs.requires_grad = True

    # Ensure scaled_inputs has the correct shape for the model
    scaled_inputs = scaled_inputs.view(-1, *input_embeddings.shape[1:])

    # Compute gradients
    outputs = model(inputs_embeds=scaled_inputs)[0]
    if target_class is not None:
        outputs = outputs[:, target_class]
    gradients = torch.autograd.grad(outputs.sum(), scaled_inputs)[0]

    # Average gradients and compute attributions
    avg_gradients = gradients[:-1].mean(dim=0)
    attributions = (input_embeddings - baseline_embeddings) * avg_gradients
    return attributions