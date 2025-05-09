#!/usr/bin/env python
# Script to train the model

import argparse
import os
import pandas as pd
import tensorflow as tf
from datetime import datetime
from app.preprocessing import preprocess_text_pairs
from app.model.translator import Translator
from app.model.tokenizers import create_or_load_tokenizers
from tensorflow.keras.preprocessing.sequence import pad_sequences

class TranslationDataset:
    def __init__(self, english_texts, itsekiri_texts, english_tokenizer, itsekiri_tokenizer, max_length=128):
        self.english_texts = english_texts
        self.itsekiri_texts = itsekiri_texts
        self.english_tokenizer = english_tokenizer
        self.itsekiri_tokenizer = itsekiri_tokenizer
        self.max_length = max_length

    def prepare_dataset(self):
        # Convert texts to sequences using Keras tokenizer
        src_sequences = self.english_tokenizer.texts_to_sequences(self.english_texts)
        tgt_sequences = self.itsekiri_tokenizer.texts_to_sequences(self.itsekiri_texts)
        
        # Pad sequences
        src_padded = pad_sequences(src_sequences, maxlen=self.max_length, padding='post')
        tgt_padded = pad_sequences(tgt_sequences, maxlen=self.max_length, padding='post')
        
        return tf.data.Dataset.from_tensor_slices((src_padded, tgt_padded))

def load_and_preprocess_data(data_path):
    """Load and preprocess the training data."""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, dtype={'clean_translation': str, 'new_target': str})
    
    # Extract English and Itsekiri text pairs
    english_texts = df['clean_translation'].tolist()
    itsekiri_texts = df['new_target'].tolist()
    
    # Preprocess the text pairs
    return preprocess_text_pairs(english_texts, itsekiri_texts)

class Trainer:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')

    @tf.function
    def train_step(self, inp, targ):
        loss = tf.constant(0.0)
        batch_size = tf.shape(inp)[0]
        seq_len = tf.shape(targ)[1]
        
        with tf.GradientTape() as tape:
            enc_hidden = self.model.encoder.initialize_hidden_state(batch_size=batch_size)
            enc_output, enc_hidden = self.model.encoder(inp, enc_hidden)
            
            dec_hidden = enc_hidden
            dec_input = tf.fill([batch_size, 1], 2)  # start token
            
            # Define loop vars and their shape invariants before the loop
            loop_vars = (
                tf.constant(1),  # iterator
                loss,
                dec_hidden,
                dec_input
            )
            
            shape_invariants = (
                tf.TensorShape([]),  # iterator shape
                tf.TensorShape([]),  # loss shape
                tf.TensorShape([None, None]),  # dec_hidden shape
                tf.TensorShape([None, 1])  # dec_input shape
            )
            
            # Custom training loop using while_loop
            def condition(t, *_):
                return tf.less(t, seq_len)
                
            def body(t, loss, dec_hidden, dec_input):
                predictions, new_dec_hidden = self.model.decoder(dec_input, dec_hidden)
                predictions = tf.reshape(predictions, [-1, tf.shape(predictions)[-1]])
                
                step_loss = self.loss_fn(targ[:, t], predictions)
                mask = tf.cast(tf.math.logical_not(tf.math.equal(targ[:, t], 0)), dtype=tf.float32)
                loss += tf.reduce_mean(step_loss * mask)
                
                new_dec_input = tf.expand_dims(targ[:, t], 1)
                return t + 1, loss, new_dec_hidden, new_dec_input
            
            # Run the training loop
            _, final_loss, _, _ = tf.while_loop(
                condition,
                body,
                loop_vars,
                shape_invariants=shape_invariants
            )
            
            # Calculate batch loss
            batch_loss = final_loss / tf.cast(seq_len - 1, dtype=tf.float32)
        
        # Get and apply gradients
        variables = self.model.encoder.trainable_variables + self.model.decoder.trainable_variables
        gradients = tape.gradient(batch_loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        
        return batch_loss

def train_model(
    data_path, 
    epochs=20, 
    batch_size=64, 
    learning_rate=5e-4, 
    hidden_size=256, 
    dropout=0.1,
    save_dir='models',
    device='cpu'
):
    """Train the Itsekiri-English translator model."""
    # Load and preprocess data
    english_texts, itsekiri_texts = load_and_preprocess_data(data_path)
    
    # Create or load tokenizers
    itsekiri_tokenizer, english_tokenizer = create_or_load_tokenizers(itsekiri_texts, english_texts)
    
    # Get vocabulary sizes
    eng_vocab_size = len(english_tokenizer.word_index) + 1  # +1 for padding token (0)
    its_vocab_size = len(itsekiri_tokenizer.word_index) + 1  # +1 for padding token (0)
    
    print(f"English vocabulary size: {eng_vocab_size}")
    print(f"Itsekiri vocabulary size: {its_vocab_size}")
    
    # Create dataset
    dataset = TranslationDataset(english_texts, itsekiri_texts, english_tokenizer, itsekiri_tokenizer)
    train_dataset = dataset.prepare_dataset().shuffle(10000).batch(batch_size)
    
    # Initialize the model
    config = {
        'embedding_dim': 256,
        'units': hidden_size,
        'batch_size': batch_size
    }
    model = Translator(config)
    model.build_model(eng_vocab_size, its_vocab_size)
    
    # Set the tokenizers
    model.inp_lang_tokenizer = english_tokenizer
    model.targ_lang_tokenizer = itsekiri_tokenizer
    
    # Training parameters
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    
    # Initialize trainer
    trainer = Trainer(model, optimizer, loss_fn)
    
    # Training loop
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        trainer.train_loss.reset_state()  # Changed from reset_states to reset_state
        
        for batch, (inp, targ) in enumerate(train_dataset):
            batch_loss = trainer.train_step(inp, targ)
            trainer.train_loss(batch_loss)
            
            if batch % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch}, Loss: {trainer.train_loss.result():.4f}")
        
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {trainer.train_loss.result():.4f}")
    
    # Save the trained model
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(save_dir, f"itsekiri_translator_{timestamp}")
    model.save_model(model_path)
    
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
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training (cpu or cuda)')
    
    args = parser.parse_args()
    
    train_model(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        save_dir=args.save_dir,
        device=args.device
    )