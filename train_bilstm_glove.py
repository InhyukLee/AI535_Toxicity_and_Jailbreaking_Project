import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.vocab import Vocab
from utils.embeddings import create_embedding_matrix
from utils.evaluate import evaluate_BG_model, print_result_metrics
from models.classifier import BiLSTM_Glove_Model
from utils.dataset import GloveLoadDataset

# --- Hyperparameters ---
TRAIN_CSV = "data/toxic-chat_annotation_train.csv"
TEST_CSV = "data/toxic-chat_annotation_test.csv"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GLOVE_PATH = "data/glove.6B.300d.txt"
BATCH_SIZE = 32
EPOCHS = 5


def main():
    # 1. Data Preparation
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # Create custom dictionary based on the words in our training data.
    v = Vocab(min_freq=2)
    v.build_vocab(train_df['user_input'])

    # Load the pre-trained GloVe and match them to the dictionary.
    weights = create_embedding_matrix(v.word2idx, GLOVE_PATH)

    # 2. Initialize the Model
    model = BiLSTM_Glove_Model(v.vocab_size, 300, 128, 2, 2, 0.5, weights).to(DEVICE)

    # set loss function
    criterion = nn.BCEWithLogitsLoss() 

    # set optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 3. Load Data
    train_loader = DataLoader(GloveLoadDataset(train_df, v), batch_size=BATCH_SIZE, shuffle=True)

    # 4. Train the model
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for texts, labels in train_loader:
            # Move the data to the GPU
            texts, labels = texts.to(DEVICE), labels.to(DEVICE)
            
            # Reset optimizer so it doesn't remember old errors
            optimizer.zero_grad()
            
            # Forward Pass
            outputs, _ = model(texts) 
            
            # Calculate Loss
            loss = criterion(outputs, labels)
            
            # Backpropagation
            loss.backward()
            
            # Apply the changes
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss/len(train_loader):.4f}")

    # 5. Evaluation
    test_loader = DataLoader(GloveLoadDataset(test_df, v), batch_size=32, shuffle=False)

    print("\nRunning Evaluation on Test Set with threshold...")
    metrics = evaluate_BG_model(model, test_loader, DEVICE, threshold=0.2)
    print_result_metrics(metrics)

    # 6. Save the model's weights
    torch.save(model.state_dict(), "models/glove_weight.pth")
    print("Weights saved successfully.")

if __name__ == "__main__":
    main()