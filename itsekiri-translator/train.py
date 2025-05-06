#!/usr/bin/env python
# Script to train the model

import argparse
import os
import pandas as pd
import torch
from datetime import datetime
from app.preprocessing import preprocess_text_pairs
from app.model.translator import ItsekiriTranslator
from app.model.tokenizers import create_or_load_tokenizers

def train_model(
    data_path, 
    epochs=20, 
    batch_size=64, 
    learning_rate=5e-4, 
    hidden_size=256, 
    dropout=0.1,
    save_dir='models'
):
    """Train the Itsekiri-English translator model."""
    print(f"Loading data from {"itsekiri-translator\data\clean_itsekiri.csv"}...")
    df = pd.read_csv("itsekiri-translator\data\clean_itsekiri.csv")
    
    # Extract English and Itsekiri text pairs
    english_texts = df['english'].tolist()
    itsekiri_texts = df['itsekiri'].tolist()
    
    # Preprocess the text pairs
    english_texts, itsekiri_texts = preprocess_text_pairs(english_texts, itsekiri_texts)
    
    # Create or load tokenizers
    english_tokenizer, itsekiri_tokenizer = create_or_load_tokenizers(english_texts, itsekiri_texts)
    
    # Initialize the model
    model = ItsekiriTranslator(
        src_vocab_size=len(english_tokenizer),
        tgt_vocab_size=len(itsekiri_tokenizer),
        hidden_size=hidden_size,
        dropout=dropout
    )
    
    # Training parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is the padding index
    
    # Training loop
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # In a real implementation, we would use DataLoader and batching
        # Here's a simplified approach
        for idx in range(0, len(english_texts), batch_size):
            batch_english = english_texts[idx:idx + batch_size]
            batch_itsekiri = itsekiri_texts[idx:idx + batch_size]
            
            # Tokenize and convert to tensors
            src_tokens = english_tokenizer.batch_encode_plus(
                batch_english, padding=True, return_tensors='pt'
            )
            tgt_tokens = itsekiri_tokenizer.batch_encode_plus(
                batch_itsekiri, padding=True, return_tensors='pt'
            )
            
            # Forward pass and loss calculation
            optimizer.zero_grad()
            outputs = model(src_tokens['input_ids'], tgt_tokens['input_ids'][:, :-1])
            
            # Reshape for loss calculation
            outputs = outputs.view(-1, outputs.size(-1))
            targets = tgt_tokens['input_ids'][:, 1:].reshape(-1)
            
            # Calculate loss, backward pass, and optimization step
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / (len(english_texts) // batch_size)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Save the trained model
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(save_dir, f"itsekiri_translator_{timestamp}.pt")
    
    # Save model and tokenizers
    torch.save({
        'model_state_dict': model.state_dict(),
        'english_tokenizer': english_tokenizer,
        'itsekiri_tokenizer': itsekiri_tokenizer,
        'model_config': {
            'hidden_size': hidden_size,
            'dropout': dropout
        }
    }, model_path)
    
    print(f"Model saved to {model_path}")
    return model, english_tokenizer, itsekiri_tokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Itsekiri-English translator model')
    parser.add_argument('--data', type=str, default='data/clean_itsekiri.csv', help='Path to the training data CSV')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=256, help='Model hidden size')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--save_dir', type=str, default='models', help='Directory to save the model')
    
    args = parser.parse_args()
    
    train_model(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        save_dir=args.save_dir
    )