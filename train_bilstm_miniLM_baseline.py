import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from utils.dataset import TransformersLoadDataset
from utils.embeddings import TransformerFeatureExtractor
from utils.evaluate import evaluate_TFF_model, print_result_metrics
from models.classifier import BiLSTM_MiniML_NoAttention_Baseline 
from train_bilstm_miniLM import MultitaskFocalLoss

# Hyperparameters - Keep these EXACTLY the same for a fair comparison
TRAIN_CSV = "data/toxic-chat_annotation_train.csv" 
TEST_CSV = "data/toxic-chat_annotation_test.csv"
BATCH_SIZE = 32
EPOCHS = 8
HEAD_LR = 1e-3         
TRANSFORMER_LR = 2e-5  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # 1. Load Data
    train_dataset = TransformersLoadDataset(TRAIN_CSV, max_len=256)
    test_dataset = TransformersLoadDataset(TEST_CSV, max_len=256)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
    # 2. Initialize the Model
    extractor = TransformerFeatureExtractor() 
    
    # Unfreeze the same layers for consistency
    for param in extractor.model.encoder.layer[-3:].parameters():
        param.requires_grad = True

    # CHANGE: Initialize the Baseline model instead of the final one
    model = BiLSTM_MiniML_NoAttention_Baseline(embedding_dim=384, hidden_dim=256).to(DEVICE)
    
    # Loss function and weights remain the same
    weights = torch.tensor([2.0, 5.0]).to(DEVICE)
    criterion = MultitaskFocalLoss(weights=weights)

    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': HEAD_LR},
        {'params': extractor.model.parameters(), 'lr': TRANSFORMER_LR}
    ], weight_decay=0.01)

    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps), 
        num_training_steps=total_steps
    )

    # 3. Training Loop
    print(f"Starting BASELINE (No-Attention) training on {DEVICE}...")
    for epoch in range(EPOCHS):
        model.train()
        extractor.model.train() 
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            smoothed_labels = labels * 0.9 + 0.05 

            optimizer.zero_grad()
            features = extractor.get_features(input_ids, mask)
            
            # The baseline returns dummy weights, so it won't break the return signature
            logits, _ = model(features)
            loss = criterion(logits, smoothed_labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Baseline Avg Loss: {avg_loss:.4f}")

    # 4. Evaluation
    print("\nBaseline training complete. Running evaluation...")
    results = evaluate_TFF_model(model, extractor, test_loader, DEVICE)
    print_result_metrics(results)

    # 5. Save baseline weights separately
    torch.save(model.state_dict(), "models/baseline_no_attention.pth")
    print("Baseline weights saved to models/baseline_no_attention.pth")

if __name__ == "__main__":
    main()