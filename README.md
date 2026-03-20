# LLM Safety Guardrail: Multi-Task Toxicity and Jailbreak Detection
This project implements a deep learning framework designed to detect Toxicity and Jailbreaking attempts in Large Language Model (LLM) prompts. It was developed as a final project for a Master’s of Computer Science, focusing on AI safety and the robust classification of harmful user inputs.

## Project Overview
As LLMs are increasingly integrated into applications, they face risks from "jailbreaking" (prompts designed to bypass safety filters) and "toxicity" (harmful or biased content). This repository provides a multi-task pipeline to identify these threats using a combination of Transformer-based embeddings and sequential neural networks.

## Model Architectures
The project compares three distinct stages of model evolution to establish a performance baseline:
| Model | Embedding | Sequence Model | Core Components | 
| ----------- | ----------- | ----------- | ----------- |
| Baseline 1 | GloVe (Static) | Bi-LSTM | 300d word vectors + PyTorch Bi-LSTM |
| Baseline 2 | MiniLM (Dynamic) | Bi-LSTM | Transformer features without Attention | 
| Final Model | MiniLM (Dynamic) | Bi-LSTM + Attention | Custom Attention mechanism + unfreezing last 3 layers |

## Performance & Evaluation
The system utilizes an Optimal Threshold Search during the evaluation phase to maximize the F1-score for both categories independently.

### Evaluation Metrics
* F1-Score / Precision / Recall: Primary metrics for imbalanced safety data.
* AUROC / AUPRC: Used to evaluate the model's ranking ability and performance across all possible thresholds.

## Getting Started
### 1. Prerequisites
Ensure you have Python installed. Install the required libraries:
```
pip install -r requirements.txt
```
### 2. Data Preparation
Place your dataset files in the data/ directory:
* toxic-chat_annotation_train.csv
* toxic-chat_annotation_test.csv
* glove.6B.300d.txt

### 3. Training & Evaluation
You can run the training scripts for the different architectures:
```
# Train GloVe Baseline
python train_bilstm_glove.py

# Train Final Transformer + Attention Model
python train_bilstm_miniLM.py

# Run comprehensive evaluation and comparison
python evaluation.py
```
You can run the evaluation scripts:
```
# Compare Three Different Model
python evaluation.py

# Show Example Case
python case_extract.py
```
