import re
from collections import Counter
import torch

class Vocab:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        # <PAD>: Fill empty space if a sentence is too short.
        # <UNK>: Used for nuknown words.
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.vocab_size = 2

    def build_vocab(self, sentences):
        # 1. Count how many times every word appears in your toxic-chat data.
        word_counts = Counter()
        for sentence in sentences:
            tokens = re.findall(r'\w+', str(sentence).lower())
            word_counts.update(tokens)
        
        # 2. Only keep words that appear at least 'min_freq' (2) times.
        for word, count in word_counts.items():
            if count >= self.min_freq:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1

    def encode(self, sentence, max_len=100):
        # 3. Turn a real sentence into a list of numbers (IDs).
        tokens = re.findall(r'\w+', str(sentence).lower())
        
        # If the word is in the dictionary, use its ID. 
        # If it's a new word, use ID 1 (<UNK>).
        encoded = [self.word2idx.get(t, 1) for t in tokens]
        
        # 4. Make sure every sentence is exactly 'max_len' long.
        if len(encoded) < max_len:
            # If it's too short, add Zeros (<PAD>) to the end.
            encoded += [0] * (max_len - len(encoded))
        else:
            # If it's too long, cut it off (Truncate).
            encoded = encoded[:max_len]
            
        return torch.tensor(encoded)