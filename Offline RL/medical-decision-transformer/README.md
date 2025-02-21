# Medical Decision Transformer

Decision Transformer implementation for sepsis treatment optimization using offline reinforcement learning with MIMIC-III dataset.

## Overview

This project implements an offline reinforcement learning approach using Decision Transformers to optimize sepsis treatment decisions. The model learns optimal treatment strategies from historical patient data in the MIMIC-III database, focusing on three key interventions:
- Antibiotics administration
- Fluid resuscitation
- Vasopressor usage

## Model Architecture

### ImprovedDecisionTransformer
- Token embedding with layer normalization
- Positional encoding
- Multi-head attention with transformer blocks
- Action, state, and return prediction heads
- Dropout and residual connections

### Key Components
- `TransformerBlock`: Custom transformer implementation
- `PositionalEncoding`: Sinusoidal position encoding
- `DecisionTransformerDataset`: Dataset handling
- `ImprovedTrainer`: Training with mixed precision

## Data Processing

The `SepsisDataProcessor` class handles:
- MIMIC-III data loading and preprocessing
- Vital signs and lab results processing
- Treatment data aggregation
- SOFA score calculation
- Data normalization

## Training Features

- Cross-validation framework
- Mixed precision training (using torch.cuda.amp)
- Early stopping
- Gradient clipping
- AdamW optimizer with weight decay


## Evaluation

The `DecisionTransformerEvaluator` provides:
- Action prediction for given states
- Trajectory evaluation
- Return correlation analysis
- MSE calculation

## License

MIT License

## Citation

If you use this code in your research, please cite:
