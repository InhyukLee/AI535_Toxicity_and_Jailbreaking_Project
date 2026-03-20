import numpy as np
import torch
from transformers import AutoModel

# --- 1. The GloVe "Dictionary" Maker ---
# This function creates a table where every word has its own list of 300 numbers
def create_embedding_matrix(word2idx, glove_path, embed_dim=300):
    vocab_size = len(word2idx)
    
    # Giving every word a set of random numbers.
    embedding_matrix = np.random.uniform(-0.05, 0.05, (vocab_size, embed_dim))
    
    found_count = 0
    print(f"Loading GloVe vectors from {glove_path}...")
    
    try:
        # Open the GloVe file and look for the words in our dataset.
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                # Replace the random numbers with the official GloVe data.
                if word in word2idx:
                    vector = np.asarray(values[1:], dtype='float32')
                    embedding_matrix[word2idx[word]] = vector
                    found_count += 1
                    
        print(f"Finished! Found embeddings for {found_count}/{vocab_size} words.")
    except FileNotFoundError:
        # If the file is missing, learn from scratch.
        print(f"Warning: {glove_path} not found. Using random initialization.")
    
    # Add the Padding index
    embedding_matrix[0] = np.zeros(embed_dim)
    
    return torch.from_numpy(embedding_matrix).float()

# --- 2. The MiniLM Feature Extractor ---
class TransformerFeatureExtractor:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        # Use the GPU if available, otherwise use the CPU.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the MiniLM model from the internet (or local folder).
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        # Freeze the parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def get_features(self, input_ids, attention_mask):
        with torch.no_grad():
            # Turn the list of Word IDs into a 3D block of Context Vectors.
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state