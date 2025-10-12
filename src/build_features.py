import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_text_embeddings(text_list, model, tokenizer, device, batch_size=64):
    """
    Generates numerical embeddings for a list of texts using a transformer model.
    """
    all_embeddings = []
    print("Generating text embeddings in batches...")
    
    for i in range(0, len(text_list), batch_size):
        batch_texts = text_list[i:i + batch_size]
        
        encoded_input = tokenizer(
            batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=128
        )
        encoded_input = {key: val.to(device) for key, val in encoded_input.items()}

        with torch.no_grad():
            model_output = model(**encoded_input)

        embeddings = model_output.last_hidden_state.mean(dim=1)
        all_embeddings.append(embeddings.cpu().numpy())
        
        print(f"Processed batch {i // batch_size + 1} / {len(text_list) // batch_size + 1}")
        
    return np.concatenate(all_embeddings, axis=0)

def main():
    """
    Main function to load data, generate text embeddings for both train and test sets, and save them.
    """
    print("--- Starting Feature Generation: Text Embeddings ---")

    # --- 1. Load Transformer Model (Load once, use for both) ---
    print("Loading DistilBERT model and tokenizer...")
    model_name = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Using device: {device}")

    # --- 2. Process Training Data ---
    print("\n--- Processing TRAINING data ---")
    train_data_path = 'data/raw/train.csv'
    train_output_path = 'data/processed/text_embeddings_distilbert.npy'

    print(f"Loading data from {train_data_path}...")
    train_df = pd.read_csv(train_data_path)
    train_content_list = train_df['catalog_content'].fillna("").tolist()
    
    train_features = get_text_embeddings(train_content_list, model, tokenizer, device)
    
    print("\nTraining embedding generation complete!")
    print(f"Shape of the final train features array: {train_features.shape}")
    
    np.save(train_output_path, train_features)
    print(f"✅ Train embeddings successfully saved to: {train_output_path}")

    # --- 3. Process Test Data ---
    print("\n--- Processing TEST data ---")
    test_data_path = 'data/raw/test.csv'
    test_output_path = 'data/processed/test_text_embeddings_distilbert.npy'

    print(f"Loading data from {test_data_path}...")
    test_df = pd.read_csv(test_data_path)
    test_content_list = test_df['catalog_content'].fillna("").tolist()

    test_features = get_text_embeddings(test_content_list, model, tokenizer, device)

    print("\nTest embedding generation complete!")
    print(f"Shape of the final test features array: {test_features.shape}")

    np.save(test_output_path, test_features)
    print(f"✅ Test embeddings successfully saved to: {test_output_path}")

if __name__ == '__main__':
    main()