import torch
import torch.nn as nn
import torch.nn.functional as F

# Model + Bi-LSTM
class BiLSTM_Glove_Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout, weights):
        super(BiLSTM_Glove_Model, self).__init__()
        # Load the pre-trained GloVe word
        self.embedding = nn.Embedding.from_pretrained(weights, freeze=False)
        
        # Bi-LSTM: Looks at the sentence from left-to-right AND right-to-left
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                           bidirectional=True, dropout=dropout, batch_first=True)
        
        self.attention_weights = nn.Parameter(torch.Tensor(hidden_dim * 2, 1))
        nn.init.xavier_uniform_(self.attention_weights)
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def attention_net(self, lstm_output):
        # Calculate an importance score for every single word in the sentence.
        scores = torch.matmul(lstm_output, self.attention_weights) 
        soft_attn_weights = F.softmax(scores, dim=1) 
        
        # Multiply the words by their importance scores and sum them up
        context = torch.sum(soft_attn_weights * lstm_output, dim=1) 
        return context, soft_attn_weights

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, _ = self.lstm(embedded)
        
        # Extract the most important features using the attention network
        attn_output, weights = self.attention_net(output)
        
        return self.fc(attn_output), weights

# MiniLM + Bi-LSTM (No Attention)
class BiLSTM_MiniML_NoAttention_Baseline(nn.Module):
    def __init__(self, embedding_dim=384, hidden_dim=256, num_layers=2):
        super(BiLSTM_MiniML_NoAttention_Baseline, self).__init__()
        # Bi-LSTM setup
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, 
                           batch_first=True, bidirectional=True, dropout=0.3)
        
        # Stability layers
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.fc_mid = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Multi-Sample Dropout (keeping this consistent for fair comparison)
        self.dropouts = nn.ModuleList([nn.Dropout(0.2 + 0.05 * i) for i in range(5)])
        self.fc_out = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        # 1. Pass through Bi-LSTM
        lstm_out, _ = self.lstm(x)
        
        # 2. Global Average Pooling
        pooled_out = torch.mean(lstm_out, dim=1) 
        
        # 3. LayerNorm for stability
        x = self.layer_norm(pooled_out)
        
        # 4. Final Classification
        x = F.relu(self.fc_mid(x))
        
        # 5. Average across dropouts
        logits = torch.mean(torch.stack([
            self.fc_out(drop(x)) for drop in self.dropouts
        ], dim=0), dim=0)
        
        dummy_weights = torch.zeros(x.size(0), 1) 
        return logits, dummy_weights

# Attention Layer for MiniLM + Bi-LSTM Model
class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        # This small neural network learns relevance patterns.
        self.attn_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(), # Tanh keeps the scores balanced between -1 and 1
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, lstm_output):
        # Calculate how much attention each word deserves
        scores = self.attn_net(lstm_output) 
        weights = F.softmax(scores, dim=1) 
        
        # Combine all words into one context vector based on weights
        context = torch.sum(weights * lstm_output, dim=1) 
        return context, weights
    
# MiniLM + Bi-LSTM
class BiLSTM_MiniML_Model(nn.Module):
    def __init__(self, embedding_dim=384, hidden_dim=256, num_layers=2):
        super(BiLSTM_MiniML_Model, self).__init__()
        # Processes the dense MiniLM vectors using a two-way LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, 
                           batch_first=True, bidirectional=True, dropout=0.3)
        
        # Use custom AttentionLayer to focus on suspicious words
        self.attention = AttentionLayer(hidden_dim * 2)
        
        # LayerNorm.
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
        self.fc_mid = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Multi-Sample Dropout
        self.dropouts = nn.ModuleList([nn.Dropout(0.2 + 0.05 * i) for i in range(5)])
        self.fc_out = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        # 1. Pass the prompt through the Bi-LSTM
        lstm_out, _ = self.lstm(x)
        
        # 2. Keep the numbers stable before the attention step
        lstm_out = self.layer_norm(lstm_out)
        
        # 3. Identify the most dangerous tokens in the sequence
        context, weights = self.attention(lstm_out)
        
        # 4. Prepare the data for the final classification
        x = F.relu(self.fc_mid(context))
        
        # 5. Get the final "Toxicity" and "Jailbreak" scores by averaging across all 5 dropout samples.
        logits = torch.mean(torch.stack([
            self.fc_out(drop(x)) for drop in self.dropouts
        ], dim=0), dim=0)
        
        return logits, weights