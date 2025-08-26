import unittest
import pandas as pd
import numpy as np
from market_pattern_analyzer import MarketPatternAnalyzer

class TestMarketPatternAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = MarketPatternAnalyzer()
        
    def test_normalize_pattern(self):
        # Create sample pattern
        data = pd.DataFrame({
            'close': [100, 110, 105, 115]
        })
        normalized = self.analyzer.normalize_pattern(data)
        self.assertEqual(len(normalized), 4)
        self.assertAlmostEqual(min(normalized), 0)
        self.assertAlmostEqual(max(normalized), 1)
        
    def test_calculate_similarity(self):
        pattern1 = np.array([0, 0.5, 1, 0.5])
        pattern2 = np.array([0, 0.5, 1, 0.5])
        similarity = self.analyzer.calculate_similarity(pattern1, pattern2)
        self.assertAlmostEqual(similarity, 1.0)
        
    def test_identify_significant_moves(self):
        data = pd.DataFrame({
            'close': [100, 126, 124, 150]
        })
        moves = self.analyzer.identify_significant_moves(data)
        self.assertEqual(len(moves), 2)  # Should identify two significant moves
        
    def test_metrics_calculation(self):
        self.analyzer.predictions = [
            {'actual': 'up', 'predicted': 'up'},
            {'actual': 'down', 'predicted': 'down'},
            {'actual': 'up', 'predicted': 'down'}
        ]
        metrics = self.analyzer.calculate_metrics()
        self.assertAlmostEqual(metrics['accuracy'], 2/3)
        self.assertAlmostEqual(metrics['win_rate'], 2/3)
        
if __name__ == '__main__':
    unittest.main()