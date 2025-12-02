import unittest
from app import app  # Import your Flask app instance

class TestModelAppIntegration(unittest.TestCase):
    def setUp(self):
        app.testing = True
        self.client = app.test_client()

    def test_model_app_integration(self):
        # Valid test input that should work with the trained model
        form_data = {
            'temperature': '275.15',  # Kelvin
            'pressure': '1013',       # hPa
            'humidity': '85',         # %
            'wind_speed': '3.6',      # m/s
            'wind_deg': '180',        # degrees
            'rain_1h': '0',           # mm
            'rain_3h': '0',           # mm
            'snow': '0',              # mm
            'clouds': '20'            # %
        }
        response = self.client.post('/', data=form_data)

        # Ensure response status is OK
        self.assertEqual(response.status_code, 200)

        # Decode HTML response
        html_text = response.data.decode('utf-8').lower()

        # Ensure that the result page includes a weather prediction
        valid_classes = [
            'clear', 'cloudy', 'drizzly', 'foggy', 'hazey',
            'misty', 'rainy', 'smokey', 'thunderstorm'
        ]
        found = any(weather in html_text for weather in valid_classes)
        self.assertTrue(found, "Prediction not found in valid weather classes.")

        # Ensure that the result page includes latency information
        self.assertIn('ms', html_text, "Latency information missing from response.")

if __name__ == '__main__':
    unittest.main()
