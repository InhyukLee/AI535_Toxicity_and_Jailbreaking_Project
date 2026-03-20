# LLM Safety Guardrail: Multi-Task Toxicity and Jailbreak Detection
This project implements a deep learning framework designed to detect Toxicity and Jailbreaking attempts in Large Language Model (LLM) prompts. It was developed as a final project for a Master’s of Computer Science, focusing on AI safety and the robust classification of harmful user inputs.
## Project Overview
As LLMs are increasingly integrated into applications, they face risks from "jailbreaking" (prompts designed to bypass safety filters) and "toxicity" (harmful or biased content). This repository provides a multi-task pipeline to identify these threats using a combination of Transformer-based embeddings and sequential neural networks.
## Model Architectures
The project compares three distinct stages of model evolution to establish a performance baseline:\
| Model | Embedding | Sequence Model | Core Components | 
| ----------- | ----------- | ----------- | ----------- |
| Baseline 1 | GloVe (Static) | Bi-LSTM | 300d word vectors + PyTorch Bi-LSTM |
| Baseline 2 | MiniLM (Dynamic) | Bi-LSTM | Transformer features without Attention | 
| Final Model | MiniLM (Fine-tuned) | Bi-LSTM + Attention | Custom Attention mechanism + unfreezing last 3 layers |
