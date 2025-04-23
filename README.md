# Integrated Hessians Library

This library provides an implementation of the Integrated Hessians method for explaining predictions of transformer-based models, such as RoBERTa, in natural language processing tasks.

## Features
- Compute Integrated Hessians for feature interaction analysis.
- Support for RoBERTa and other transformer-based models.
- Example usage with IMDb sentiment analysis dataset.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd integrated_gradients
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Example Usage

Run the example script:
```bash
python examples/example_usage.py
```

### Library Usage

1. Import the library:
   ```python
   from integrated_gradients.integrated_hessians import integrated_hessians
   from integrated_gradients.model_utils import load_model_and_tokenizer
   ```

2. Load a model and tokenizer:
   ```python
   tokenizer, model = load_model_and_tokenizer('cardiffnlp/twitter-roberta-base-sentiment')
   ```

3. Compute Integrated Hessians:
   ```python
   ih_matrix = integrated_hessians(model, input_ids, baseline_ids, steps=20, target_class=1)
   print(ih_matrix)
   ```

## Project Structure
- `integrated_gradients/`: Library code.
  - `integrated_hessians.py`: Implementation of Integrated Hessians.
  - `model_utils.py`: Utilities for loading models and tokenizers.
- `examples/`: Example scripts.
  - `example_usage.py`: Example usage of the library.

## License
This project is licensed under the MIT License.