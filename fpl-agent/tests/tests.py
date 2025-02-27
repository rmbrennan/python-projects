import unittest
from unittest.mock import patch, Mock
import pandas as pd
from utils.utils import DataFetcher

class TestDataFetcher(unittest.TestCase):
    @patch('utils.utils.requests.get')
    def test_load_modeling_data(self, mock_get):
        # Mock the API response
        mock_response = Mock()
        mock_response.json.return_value = {
            'elements': [
                {
                    'id': 1,
                    'web_name': 'Player 1',
                    'element_type': 1,
                    'now_cost': 100,
                    'ep_next': 5.0,
                    'minutes': 1000,
                    'form': '5.0',
                    'points_per_game': '4.5',
                    'selected_by_percent': '10.0',
                    'transfers_in': 1000,
                    'transfers_out': 500,
                    'influence': '100.0',
                    'creativity': '50.0',
                    'threat': '75.0',
                    'ict_index': '225.0',
                    'team': 1
                }
            ]
        }
        mock_get.return_value = mock_response

        # Create an instance of DataFetcher
        data_fetcher = DataFetcher()

        # Call the load_modeling_data method
        result_df = data_fetcher.load_modeling_data()

        # Define the expected DataFrame
        expected_df = pd.DataFrame([{
            'id': 1,
            'web_name': 'Player 1',
            'element_type': 1,
            'now_cost': 100,
            'ep_next': 5.0,
            'total_season_minutes': 1000,
            'form': '5.0',
            'points_per_game': '4.5',
            'selected_by_percent': '10.0',
            'transfers_in': 1000,
            'transfers_out': 500,
            'influence': '100.0',
            'creativity': '50.0',
            'threat': '75.0',
            'ict_index': '225.0',
            'team': 1
        }])

        # Assert that the result matches the expected DataFrame
        pd.testing.assert_frame_equal(result_df, expected_df)

if __name__ == '__main__':
    unittest.main()