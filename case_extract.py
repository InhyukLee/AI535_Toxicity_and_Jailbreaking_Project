import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from transformers import AutoTokenizer

from utils.dataset import TransformersLoadDataset
from utils.embeddings import TransformerFeatureExtractor
from models.classifier import BiLSTM_MiniML_Model

TEST_CSV = "data/toxic-chat_annotation_test.csv"
FINAL_WEIGHTS = "models/transformer_weight_final.pth"
FINE_TUNED_MINILM = "models/fine_tuned_minilm"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def get_top_10_tokens(input_ids, weights):
    """Matches attention weights to tokens and returns the top 10."""
    # Convert IDs back to tokens using the MiniLM tokenizer
    tokens = TOKENIZER.convert_ids_to_tokens(input_ids)
    weights = weights.squeeze().cpu().numpy()
    
    # Filter out special characters ([PAD], [CLS], [SEP]) for a cleaner report
    pairs = [(t, w) for t, w in zip(tokens, weights) if t not in ['[PAD]', '[CLS]', '[SEP]']]
    
    # Sort by the attention weight value
    return sorted(pairs, key=lambda x: x[1], reverse=True)[:10]

def main():
    # 1. Setup Model and Data
    dataset = TransformersLoadDataset(TEST_CSV, max_len=256)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    extractor = TransformerFeatureExtractor(model_name=FINE_TUNED_MINILM)
    model = BiLSTM_MiniML_Model(embedding_dim=384, hidden_dim=256).to(DEVICE)
    model.load_state_dict(torch.load(FINAL_WEIGHTS, map_location=DEVICE))
    model.eval()

    cases = {
        "True Positive (Toxic)": None,
        "True Positive (Jailbreak)": None,
        "False Positive (Error)": None,
        "False Negative (Missed)": None
    }

    print("Searching for qualitative examples and extracting attention words...")
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            input_ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].cpu().numpy()[0] # [Tox, Jail]
            
            # Extract features and get weights from the AttentionLayer
            features = extractor.get_features(input_ids, mask)
            logits, weights = model(features)
            
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            preds = (probs > 0.5).astype(int)
            text = dataset.df.iloc[i]['user_input']

            # Selection logic
            target_key = None
            if labels[0] == 1 and preds[0] == 1 and cases["True Positive (Toxic)"] is None:
                target_key = "True Positive (Toxic)"
            elif labels[1] == 1 and preds[1] == 1 and cases["True Positive (Jailbreak)"] is None:
                target_key = "True Positive (Jailbreak)"
            elif labels[0] == 0 and preds[0] == 1 and cases["False Positive (Error)"] is None:
                target_key = "False Positive (Error)"
            elif labels[1] == 1 and preds[1] == 0 and cases["False Negative (Missed)"] is None:
                target_key = "False Negative (Missed)"

            if target_key:
                # Extract the Top 10 tokens that triggered the attention mechanism
                top_10 = get_top_10_tokens(input_ids[0], weights[0])
                conf = probs[0] if "Toxic" in target_key or "Error" in target_key else probs[1]
                cases[target_key] = (text, conf, top_10)

            if all(v is not None for v in cases.values()):
                break

    # 2. Output the result
    print("\n" + "="*80)
    for case_type, (text, conf, top_words) in cases.items():
        print(f"TYPE: {case_type}")
        print(f"CONFIDENCE: {conf:.4f}")
        print(f"TEXT: {text[:150]}...") 
        word_list = ", ".join([f"{word} ({w:.2f})" for word, w in top_words])
        print(f"TOP 10 ATTENTION WORDS: {word_list}")
        print("-" * 80)

if __name__ == "__main__":
    main()