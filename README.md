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
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Look inside examples folder
Example usage can be found in its specified jupyter notebook


## Project Structure
- `src/`: Library code.
  - `integrated_gradients.py`: Implementation of Integrated Gradients.
  - `integrated_hessians.py`: Implementation of Integrated Hessians. integrated_hessian is approximated version of integrated_hessian_full.
  - `model_utils.py`: Utilities for loading models and tokenizers.
- `examples/`: Example scripts.
  - `integrated_hessians_demo.ipynb`: Example usage of the library.
