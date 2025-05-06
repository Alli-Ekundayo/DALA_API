import json
import unittest
from app import api
from app.config import TestingConfig

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.app = api.create_app(TestingConfig)
        self.client = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()

    def tearDown(self):
        self.app_context.pop()

    def test_health_endpoint(self):
        """Test the health check endpoint."""
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'ok')

    def test_translate_english_to_itsekiri(self):
        """Test the English to Itsekiri translation endpoint."""
        response = self.client.post(
            '/api/translate/english-to-itsekiri',
            json={'text': 'Hello, how are you?'}
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('translation', data)
        self.assertIsInstance(data['translation'], str)

    def test_translate_itsekiri_to_english(self):
        """Test the Itsekiri to English translation endpoint."""
        response = self.client.post(
            '/api/translate/itsekiri-to-english',
            json={'text': 'Omọ ọle, mẹru wa?'}
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('translation', data)
        self.assertIsInstance(data['translation'], str)

    def test_invalid_input(self):
        """Test error handling with invalid input."""
        response = self.client.post(
            '/api/translate/english-to-itsekiri',
            json={}  # Missing 'text' field
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)

    def test_model_loading_error(self):
        """Test handling when model can't be loaded."""
        # Temporarily change model path to trigger an error
        original_model_path = self.app.config['MODEL_PATH']
        self.app.config['MODEL_PATH'] = 'nonexistent_model.pt'
        
        response = self.client.post(
            '/api/translate/english-to-itsekiri',
            json={'text': 'Hello'}
        )
        self.assertEqual(response.status_code, 500)
        data = json.loads(response.data)
        self.assertIn('error', data)
        
        # Restore the original model path
        self.app.config['MODEL_PATH'] = original_model_path

if __name__ == '__main__':
    unittest.main()