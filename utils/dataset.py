import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# --- 1. Dataset for GloVe (Word-Vector based) ---
class GloveLoadDataset(Dataset):
    def __init__(self, dataframe, vocab, max_len=100):
        self.data = dataframe
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 1. Grab one row from our Excel/CSV data
        row = self.data.iloc[idx]
        text = row['user_input']
        
        # 2. Labels: We create a list of two numbers [is_toxic, is_jailbreak]
        label = torch.tensor([row['toxicity'], row['jailbreaking']], dtype=torch.float32)
        
        # 3. Convert words to numbers (Numericalize)
        tokens = self.vocab.encode(text, max_len=self.max_len)
        
        return tokens, label

# --- 2. Dataset for Transformers (MiniLM based) ---
class TransformersLoadDataset(Dataset):
    def __init__(self, csv_path, model_name="sentence-transformers/all-MiniLM-L6-v2", max_len=128):
        # Load the CSV file into a Pandas table
        self.df = pd.read_csv(csv_path)
        
        # Load the Dictionary (Tokenizer) that matches the MiniLM model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_len = max_len
        
        # Extract the columns we need
        self.texts = self.df['user_input'].values
        self.toxic_labels = self.df['toxicity'].values
        self.jailbreak_labels = self.df['jailbreaking'].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # 1. Get the text and turn it into a string
        text = str(self.texts[idx])
        
        # 2. Prepare the labels as a small vector
        labels = torch.tensor([self.toxic_labels[idx], self.jailbreak_labels[idx]], dtype=torch.float)

        # 3. The Tokenization step
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,     # Adds [CLS] at start and [SEP] at end
            max_length=self.max_len,     # Cuts off long sentences
            padding='max_length',        # Makes short sentences longer using zeros
            truncation=True,             # Ensures we don't crash on huge inputs
            return_tensors='pt'          # Returns data as PyTorch Tensors
        )

        # 4. Return the data in a dictionary format
        return {
            'input_ids': encoding['input_ids'].flatten(),           # The "word IDs"
            'attention_mask': encoding['attention_mask'].flatten(), # Tells model which words are real vs padding
            'label': labels                                         # The [0, 1] safety labels
        }