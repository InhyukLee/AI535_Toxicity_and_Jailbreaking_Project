import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from utils.vocab import Vocab
from utils.embeddings import create_embedding_matrix, TransformerFeatureExtractor
from utils.dataset import GloveLoadDataset, TransformersLoadDataset
from utils.evaluate import evaluate_BG_model, evaluate_TFF_model
from models.classifier import BiLSTM_Glove_Model, BiLSTM_MiniML_NoAttention_Baseline, BiLSTM_MiniML_Model

# Configuration
TRAIN_CSV = "data/toxic-chat_annotation_train.csv"
TEST_CSV = "data/toxic-chat_annotation_test.csv"
GLOVE_PATH = "data/glove.6B.300d.txt"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Weight paths
GLOVE_WEIGHTS = "models/glove_weight.pth"
BASELINE_WEIGHTS = "models/baseline_no_attention.pth"
FINAL_WEIGHTS = "models/transformer_weight_final.pth"
FINE_TUNED_MINILM = "models/fine_tuned_minilm" 

def format_row(model_name, category, metrics):
    return f"{model_name:<20} | {category:<10} | {metrics['f1']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['auroc']:.4f}"

def main():
    print(f"Running evaluation on {DEVICE}...\n")
    all_results = []

    # 1. Evaluate GloVe + Bi-LSTM
    print("Evaluating GloVe Baseline...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    
    # Build vocab to match training state
    v = Vocab(min_freq=2)
    v.build_vocab(train_df['user_input'])
    weights = create_embedding_matrix(v.word2idx, GLOVE_PATH)
    
    glove_model = BiLSTM_Glove_Model(v.vocab_size, 300, 128, 2, 2, 0.5, weights).to(DEVICE)
    glove_model.load_state_dict(torch.load(GLOVE_WEIGHTS, map_location=DEVICE))
    
    glove_loader = DataLoader(GloveLoadDataset(test_df, v), batch_size=32, shuffle=False)
    # Using 0.2 threshold as per final GloVe training script
    glove_res = evaluate_BG_model(glove_model, glove_loader, DEVICE, threshold=0.2)
    all_results.append(("GloVe + Bi-LSTM", glove_res))


    # 2. Setup Transformer Components
    # Load the fine-tuned extractor
    try:
        extractor = TransformerFeatureExtractor(model_name=FINE_TUNED_MINILM)
        print("Loaded fine-tuned MiniLM extractor.")
    except:
        extractor = TransformerFeatureExtractor()
        print("Loaded base MiniLM extractor (fine-tuned weights not found).")

    trans_dataset = TransformersLoadDataset(TEST_CSV, max_len=256)
    trans_loader = DataLoader(trans_dataset, batch_size=32, shuffle=False)

    # 3. Evaluate MiniLM + Bi-LSTM (No Attention)
    print("Evaluating MiniLM Baseline (No Attention)...")
    baseline_model = BiLSTM_MiniML_NoAttention_Baseline(embedding_dim=384, hidden_dim=256).to(DEVICE)
    baseline_model.load_state_dict(torch.load(BASELINE_WEIGHTS, map_location=DEVICE))
    
    baseline_res = evaluate_TFF_model(baseline_model, extractor, trans_loader, DEVICE)
    all_results.append(("MiniLM + Bi-LSTM", baseline_res))


    # 4. Evaluate Final Model (MiniLM + Bi-LSTM + Attention)
    print("Evaluating Final Model (With Attention)...")
    final_model = BiLSTM_MiniML_Model(embedding_dim=384, hidden_dim=256).to(DEVICE)
    final_model.load_state_dict(torch.load(FINAL_WEIGHTS, map_location=DEVICE))
    
    final_res = evaluate_TFF_model(final_model, extractor, trans_loader, DEVICE)
    all_results.append(("Final (Attention)", final_res))


    # 5. Print Final Comparison Table
    print("\n" + "="*85)
    print(f"{'Model Architecture':<20} | {'Category':<10} | {'F1':<6} | {'Prec':<6} | {'Recall':<6} | {'AUROC':<6}")
    print("-" * 85)
    
    for name, res in all_results:
        for cat in ['Toxicity', 'Jailbreak']:
            print(format_row(name, cat, res[cat]))
        print("-" * 85)
    print("="*85)

if __name__ == "__main__":
    main()