import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from utils.dataset import TransformersLoadDataset
from utils.embeddings import TransformerFeatureExtractor
from utils.evaluate import evaluate_TFF_model, print_result_metrics
from models.classifier import BiLSTM_MiniML_Model

# Multitask Focal Loss
class MultitaskFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, weights=None):
        super(MultitaskFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma # High gamma means "focus more on hard mistakes"
        self.weights = weights

    def forward(self, inputs, targets):
        # Calculate standard error
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        
        # This formula reduces the loss for "easy" correct guesses 
        # so they don't overwhelm the "hard" incorrect ones.
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss
        
        if self.weights is not None:
            F_loss = F_loss * self.weights
            
        return torch.mean(F_loss)

# Hyperparameters
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
    
    # Unfreezing the last 3 layers of MiniLM so that the model adapt its general English knowledge to specific "Toxic Chat" slang.
    for param in extractor.model.encoder.layer[-3:].parameters():
        param.requires_grad = True

    model = BiLSTM_MiniML_Model(embedding_dim=384, hidden_dim=256).to(DEVICE)
    
    # set loss function
    weights = torch.tensor([2.0, 5.0]).to(DEVICE)
    criterion = MultitaskFocalLoss(weights=weights)

    # set optimizer
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': HEAD_LR},
        {'params': extractor.model.parameters(), 'lr': TRANSFORMER_LR}
    ], weight_decay=0.01)

    # set scheduler
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps), 
        num_training_steps=total_steps
    )

    # 3. Train the model
    print(f"Starting final optimized training on {DEVICE}...")
    for epoch in range(EPOCHS):
        model.train()
        extractor.model.train() 
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            # Label Smoothing to prevent overfitting.
            smoothed_labels = labels * 0.9 + 0.05 

            optimizer.zero_grad()
            
            # Extract features (MiniLM context vectors)
            features = extractor.get_features(input_ids, mask)
            
            # Pass through Bi-LSTM and Attention
            logits, _ = model(features)
            loss = criterion(logits, smoothed_labels)
            
            # Backpropagation
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Apply the changes
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Avg Loss: {avg_loss:.4f}")

    # 4. Final Test
    print("\nTraining complete. Running threshold-optimized evaluation...")
    results = evaluate_TFF_model(model, extractor, test_loader, DEVICE)
    print_result_metrics(results)

    # 5. Save both the "Brain" and the "Head"
    torch.save(model.state_dict(), "models/transformer_weight_final.pth")
    extractor.model.save_pretrained("models/fine_tuned_minilm")
    print("All weights saved successfully.")

if __name__ == "__main__":
    main()