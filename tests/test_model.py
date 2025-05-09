import unittest
import torch
import os
import tempfile
from app.model.translator import ItsekiriTranslator
from app.model.tokenizers import create_or_load_tokenizers

class TestTranslatorModel(unittest.TestCase):
    def setUp(self):
        # Sample text for tokenizer testing
        self.english_texts = [
            "Hello, how are you?",
            "My name is John.",
            "I am learning Itsekiri language."
        ]
        self.itsekiri_texts = [
            "Omọ ọle, mẹru wa?",
            "Orukọ mi John.",
            "Me je kekere Itsekiri."
        ]
        
        # Create sample tokenizers
        self.english_tokenizer, self.itsekiri_tokenizer = create_or_load_tokenizers(
            self.english_texts, self.itsekiri_texts
        )
        
        # Create a small model for testing
        self.model = ItsekiriTranslator(
            src_vocab_size=len(self.english_tokenizer),
            tgt_vocab_size=len(self.itsekiri_tokenizer),
            hidden_size=64,  # Small size for testing
            dropout=0.1
        )

    def test_model_initialization(self):
        """Test that the model initializes correctly."""
        self.assertIsInstance(self.model, ItsekiriTranslator)
        
        # Check model components
        self.assertEqual(self.model.encoder.embedding_dim, 64)
        self.assertEqual(self.model.decoder.embedding_dim, 64)

    def test_model_forward_pass(self):
        """Test the model's forward pass."""
        # Encode a sample sentence
        sample_en = "Hello, how are you?"
        tokens = self.english_tokenizer.encode_plus(
            sample_en, 
            return_tensors='pt'
        )
        
        # Create a target sequence (dummy)
        tgt_tokens = torch.ones((1, 5), dtype=torch.long)
        
        # Run forward pass
        output = self.model(tokens['input_ids'], tgt_tokens)
        
        # Check output shape
        self.assertEqual(output.shape[0], 1)  # Batch size
        self.assertEqual(output.shape[1], 5)  # Sequence length
        self.assertEqual(output.shape[2], len(self.itsekiri_tokenizer))  # Vocab size

    def test_model_save_and_load(self):
        """Test saving and loading the model."""
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            model_path = tmp.name
        
        # Save the model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'english_tokenizer': self.english_tokenizer,
            'itsekiri_tokenizer': self.itsekiri_tokenizer,
            'model_config': {
                'hidden_size': 64,
                'dropout': 0.1
            }
        }, model_path)
        
        # Load the model
        checkpoint = torch.load(model_path)
        
        # Initialize a new model with the saved config
        loaded_model = ItsekiriTranslator(
            src_vocab_size=len(checkpoint['english_tokenizer']),
            tgt_vocab_size=len(checkpoint['itsekiri_tokenizer']),
            hidden_size=checkpoint['model_config']['hidden_size'],
            dropout=checkpoint['model_config']['dropout']
        )
        loaded_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Check that weights are the same
        for p1, p2 in zip(self.model.parameters(), loaded_model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))
        
        # Clean up
        os.unlink(model_path)

    def test_tokenizers(self):
        """Test the tokenizers."""
        # Test English tokenizer
        encoded = self.english_tokenizer.encode_plus(
            "Hello, world!", 
            return_tensors='pt'
        )
        self.assertIn('input_ids', encoded)
        self.assertIn('attention_mask', encoded)
        
        # Test Itsekiri tokenizer
        encoded = self.itsekiri_tokenizer.encode_plus(
            "Omọ ọle!", 
            return_tensors='pt'
        )
        self.assertIn('input_ids', encoded)
        self.assertIn('attention_mask', encoded)
        
        # Test decoding
        tokens = self.english_tokenizer.encode("Hello")
        decoded = self.english_tokenizer.decode(tokens)
        self.assertIn("Hello", decoded)

if __name__ == '__main__':
    unittest.main()