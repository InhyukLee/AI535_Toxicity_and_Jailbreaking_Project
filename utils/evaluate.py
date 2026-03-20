import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score

# --- 1. Evaluation for the GloVe Model ---
def evaluate_BG_model(model, test_loader, device, threshold=0.3):
    model.eval()
    
    all_probs = []
    all_labels = []
    
    # 'no_grad' saves memory because we don't need to calculate 
    with torch.no_grad():
        for texts, labels in test_loader:
            texts = texts.to(device)
            # Get the raw numbers (logits) from the model
            outputs, _ = model(texts) 
            
            # Sigmoid turns raw numbers into a 0.0 to 1.0 probability.
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            all_probs.append(probs)
            all_labels.append(labels.numpy())
            
    # Combine all small groups (batches) into one giant list
    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)

    categories = ['Toxicity', 'Jailbreak']
    results = {}

    for i, cat in enumerate(categories):
        cat_probs = all_probs[:, i]
        cat_labels = all_labels[:, i]
        cat_preds = (cat_probs > threshold).astype(int)
        
        # Calculate Precision, Recall, F1 Scores
        precision, recall, f1, _ = precision_recall_fscore_support(
            cat_labels, cat_preds, average='binary', zero_division=0
        )
        
        # Calculate AUROC & AUPRC
        auroc = roc_auc_score(cat_labels, cat_probs)
        auprc = average_precision_score(cat_labels, cat_probs)
        
        results[cat] = {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "auroc": auroc,
            "auprc": auprc
        }
        
    return results

# --- 2. Evaluation for the Transformer Model ---
def evaluate_TFF_model(model, extractor, data_loader, device):
    model.eval()
    extractor.model.eval()
    
    all_probs, all_labels = [], []
    
    with torch.no_grad():
        for batch in data_loader:
            # 1. Extract context-aware features using MiniLM
            features = extractor.get_features(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            
            # 2. Get the classification results from the Bi-LSTM head
            logits, _ = model(features)
            
            # 3. Store the 0-1 probabilities
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(batch['label'].cpu().numpy())

    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    
    categories = ['Toxicity', 'Jailbreak']
    results = {}

    for i, cat in enumerate(categories):
        best_f1, best_thresh = 0, 0.5
        cat_probs, cat_labels = all_probs[:, i], all_labels[:, i]
        
        # Threshold searching
        for thresh in np.arange(0.2, 0.7, 0.01):
            preds = (cat_probs > thresh).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(
                cat_labels, preds, average='binary', zero_division=0
            )
            
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        # Calculate final scores using the Best threshold
        final_preds = (cat_probs > best_thresh).astype(int)
        p, r, f, _ = precision_recall_fscore_support(cat_labels, final_preds, average='binary', zero_division=0)
        
        results[cat] = {
            "f1": f,
            "precision": p,
            "recall": r,
            "auroc": roc_auc_score(cat_labels, cat_probs),
            "auprc": average_precision_score(cat_labels, cat_probs)
        }
        print(f"Optimal {cat} Threshold: {best_thresh:.3f}")

    return results

# --- 3. Result Printer ---
def print_result_metrics(results):
    print("\n" + "="*40)
    print(f"{'Metric':<12} | {'Toxicity':<10} | {'Jailbreak':<10}")
    print("-" * 40)
    # Print F1, Precision, Recall, AUROC, and AUPRC
    for m in ["f1", "precision", "recall", "auroc", "auprc"]:
        print(f"{m.upper():<12} | {results['Toxicity'][m]:.4f}     | {results['Jailbreak'][m]:.4f}")
    print("="*40)