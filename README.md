# LLM Safety Guardrail: Multi-Task Toxicity and Jailbreak Detection\
This project implements a deep learning framework designed to detect Toxicity and Jailbreaking attempts in Large Language Model (LLM) prompts. It was developed as a final project for a Master’s of Computer Science, focusing on AI safety and the robust classification of harmful user inputs.\
## Project Overview\
As LLMs are increasingly integrated into applications, they face risks from "jailbreaking" (prompts designed to bypass safety filters) and "toxicity" (harmful or biased content). This repository provides a multi-task pipeline to identify these threats using a combination of Transformer-based embeddings and sequential neural networks.\
### Key Features\
*Multi-Task Learning: Simultaneously classifies prompts for both Toxicity and Jailbreaking using a single shared backbone.
*Hybrid Architecture: Leverages the contextual power of MiniLM combined with a Bi-LSTM and a custom Attention Layer.
*Optimized Training: Implements Multitask Focal Loss to handle class imbalance and Label Smoothing to improve model generalization.
*Explainability: Includes an attention-weight extraction tool to visualize which specific tokens (words) triggered a safety flag.
