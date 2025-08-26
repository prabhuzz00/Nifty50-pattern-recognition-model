# import pandas as pd
# import numpy as np
# from typing import List, Tuple
# from datetime import datetime
# import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class MarketPatternAnalyzer:
#     def __init__(self, min_pattern_length: int = 4, max_pattern_length: int = 10, 
#                  movement_threshold: float = 25.0, similarity_threshold: float = 0.85):
#         """
#         Initialize the Market Pattern Analyzer.
        
#         Args:
#             min_pattern_length: Minimum number of candles to consider for pattern
#             max_pattern_length: Maximum number of candles to consider for pattern
#             movement_threshold: Minimum price movement to consider significant
#             similarity_threshold: Threshold for pattern similarity (0-1)
#         """
#         self.min_pattern_length = min_pattern_length
#         self.max_pattern_length = max_pattern_length
#         self.movement_threshold = movement_threshold
#         self.similarity_threshold = similarity_threshold
#         self.patterns_memory = []
#         self.predictions = []
        
#     def load_data(self, file_path: str) -> pd.DataFrame:
#         """Load and preprocess OHLC data from CSV file."""
#         try:
#             df = pd.read_csv(file_path)
#             df['date'] = pd.to_datetime(df['date'])
#             df.set_index('date', inplace=True)
#             return df
#         except Exception as e:
#             logger.error(f"Error loading data: {e}")
#             raise
            
#     def normalize_pattern(self, pattern: pd.DataFrame) -> np.ndarray:
#         """Normalize pattern for comparison."""
#         closes = pattern['close'].values
#         min_val = closes.min()
#         max_val = closes.max()
#         if max_val - min_val == 0:
#             return np.zeros_like(closes)
#         return (closes - min_val) / (max_val - min_val)
    
#     def calculate_similarity(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
#         """Calculate similarity between two patterns using correlation."""
#         if len(pattern1) != len(pattern2):
#             return 0
#         correlation = np.corrcoef(pattern1, pattern2)[0, 1]
#         return abs(correlation) if not np.isnan(correlation) else 0
    
#     def identify_significant_moves(self, data: pd.DataFrame) -> List[Tuple[int, str]]:
#         """Identify points where price moves more than threshold."""
#         significant_moves = []
#         for i in range(len(data) - 1):
#             price_change = data['close'].iloc[i+1] - data['close'].iloc[i]
#             if abs(price_change) >= self.movement_threshold:
#                 direction = 'up' if price_change > 0 else 'down'
#                 significant_moves.append((i, direction))
#         return significant_moves
    
#     def extract_pattern(self, data: pd.DataFrame, end_idx: int, 
#                        pattern_length: int) -> pd.DataFrame:
#         """Extract pattern of given length ending at specified index."""
#         start_idx = end_idx - pattern_length + 1
#         if start_idx < 0:
#             return None
#         return data.iloc[start_idx:end_idx + 1]
    
#     def train_and_analyze(self, data: pd.DataFrame):
#         """Train the model by analyzing patterns and storing them."""
#         significant_moves = self.identify_significant_moves(data)
#         logger.info(f"Found {len(significant_moves)} significant moves")
        
#         for idx, direction in significant_moves:
#             for pattern_length in range(self.min_pattern_length, 
#                                       self.max_pattern_length + 1):
#                 pattern = self.extract_pattern(data, idx, pattern_length)
#                 if pattern is None:
#                     continue
                
#                 normalized_pattern = self.normalize_pattern(pattern)
#                 similar_patterns = [p for p in self.patterns_memory 
#                                   if len(p['pattern']) == len(normalized_pattern) and
#                                   self.calculate_similarity(p['pattern'], 
#                                                          normalized_pattern) >= 
#                                   self.similarity_threshold]
                
#                 if similar_patterns:
#                     # Make prediction based on similar patterns
#                     predicted_direction = max(set(p['direction'] for p in similar_patterns),
#                                            key=lambda x: sum(1 for p in similar_patterns 
#                                                            if p['direction'] == x))
#                     self.predictions.append({
#                         'timestamp': data.index[idx],
#                         'actual': direction,
#                         'predicted': predicted_direction,
#                         'pattern_length': pattern_length
#                     })
                
#                 # Store the pattern
#                 self.patterns_memory.append({
#                     'pattern': normalized_pattern,
#                     'direction': direction,
#                     'length': pattern_length
#                 })
    
#     def get_metrics(self) -> dict:
#         """Calculate and return performance metrics."""
#         if not self.predictions:
#             return {
#                 'total_predictions': 0,
#                 'accuracy': 0.0,
#                 'win_rate': 0.0
#             }
        
#         total = len(self.predictions)
#         correct = sum(1 for p in self.predictions 
#                      if p['actual'] == p['predicted'])
        
#         metrics = {
#             'total_predictions': total,
#             'accuracy': correct / total,
#             'win_rate': correct / total,
#             'total_patterns': len(self.patterns_memory)
#         }
        
#         return metrics

# def main():
#     # Initialize analyzer
#     analyzer = MarketPatternAnalyzer(
#         min_pattern_length=4,
#         max_pattern_length=10,
#         movement_threshold=25.0,
#         similarity_threshold=0.85
#     )
    
#     try:
#         # Load your data - replace with your file path
#         data = analyzer.load_data('NIFTY_50_minute.csv')
#         logger.info(f"Loaded {len(data)} data points")
        
#         # Train and analyze patterns
#         analyzer.train_and_analyze(data)
        
#         # Get and display metrics
#         metrics = analyzer.get_metrics()
#         logger.info("Analysis Results:")
#         logger.info(f"Total Patterns Stored: {metrics['total_patterns']}")
#         logger.info(f"Total Predictions Made: {metrics['total_predictions']}")
#         logger.info(f"Prediction Accuracy: {metrics['accuracy']:.2%}")
#         logger.info(f"Win Rate: {metrics['win_rate']:.2%}")
        
#     except Exception as e:
#         logger.error(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()

#########################################################################################

# import pandas as pd
# import numpy as np
# from typing import List, Tuple
# from datetime import datetime
# import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class MarketPatternAnalyzer:
#     def __init__(self, min_pattern_length: int = 4, max_pattern_length: int = 10, 
#                  movement_threshold: float = 25.0, similarity_threshold: float = 0.85,
#                  train_start: str = '2021-01-01 09:15:00',
#                  train_end: str = '2025-01-01 09:15:00'):
#         """
#         Initialize the Market Pattern Analyzer.
        
#         Args:
#             min_pattern_length: Minimum number of candles to consider for pattern
#             max_pattern_length: Maximum number of candles to consider for pattern
#             movement_threshold: Minimum price movement to consider significant
#             similarity_threshold: Threshold for pattern similarity (0-1)
#             train_start: Start date for training period (YYYY-MM-DD HH:MM:SS)
#             train_end: End date for training period (YYYY-MM-DD HH:MM:SS)
#         """
#         self.min_pattern_length = min_pattern_length
#         self.max_pattern_length = max_pattern_length
#         self.movement_threshold = movement_threshold
#         self.similarity_threshold = similarity_threshold
#         self.train_start = pd.to_datetime(train_start)
#         self.train_end = pd.to_datetime(train_end)
#         self.patterns_memory = []
#         self.train_predictions = []
#         self.test_predictions = []
        
#     def load_data(self, file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
#         """Load and preprocess OHLC data from CSV file, split into train and test sets."""
#         try:
#             df = pd.read_csv(file_path)
#             df['date'] = pd.to_datetime(df['date'])
#             df.set_index('date', inplace=True)
            
#             # Split data into training and testing sets
#             train_data = df[self.train_start:self.train_end]
#             test_data = df[self.train_end:]
            
#             logger.info(f"Training data period: {train_data.index.min()} to {train_data.index.max()}")
#             logger.info(f"Testing data period: {test_data.index.min()} to {test_data.index.max()}")
            
#             return train_data, test_data
#         except Exception as e:
#             logger.error(f"Error loading data: {e}")
#             raise
            
#     def normalize_pattern(self, pattern: pd.DataFrame) -> np.ndarray:
#         """Normalize pattern for comparison."""
#         closes = pattern['close'].values
#         min_val = closes.min()
#         max_val = closes.max()
#         if max_val - min_val == 0:
#             return np.zeros_like(closes)
#         return (closes - min_val) / (max_val - min_val)
    
#     def calculate_similarity(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
#         """Calculate similarity between two patterns using correlation."""
#         if len(pattern1) != len(pattern2):
#             return 0
#         correlation = np.corrcoef(pattern1, pattern2)[0, 1]
#         return abs(correlation) if not np.isnan(correlation) else 0
    
#     def identify_significant_moves(self, data: pd.DataFrame) -> List[Tuple[int, str]]:
#         """Identify points where price moves more than threshold."""
#         significant_moves = []
#         for i in range(len(data) - 1):
#             price_change = data['close'].iloc[i+1] - data['close'].iloc[i]
#             if abs(price_change) >= self.movement_threshold:
#                 direction = 'up' if price_change > 0 else 'down'
#                 significant_moves.append((i, direction))
#         return significant_moves
    
#     def extract_pattern(self, data: pd.DataFrame, end_idx: int, 
#                        pattern_length: int) -> pd.DataFrame:
#         """Extract pattern of given length ending at specified index."""
#         start_idx = end_idx - pattern_length + 1
#         if start_idx < 0:
#             return None
#         return data.iloc[start_idx:end_idx + 1]
    
#     def train(self, train_data: pd.DataFrame):
#         """Train the model using training data."""
#         significant_moves = self.identify_significant_moves(train_data)
#         logger.info(f"Found {len(significant_moves)} significant moves in training data")
        
#         for idx, direction in significant_moves:
#             for pattern_length in range(self.min_pattern_length, 
#                                       self.max_pattern_length + 1):
#                 pattern = self.extract_pattern(train_data, idx, pattern_length)
#                 if pattern is None:
#                     continue
                
#                 normalized_pattern = self.normalize_pattern(pattern)
#                 similar_patterns = [p for p in self.patterns_memory 
#                                   if len(p['pattern']) == len(normalized_pattern) and
#                                   self.calculate_similarity(p['pattern'], 
#                                                          normalized_pattern) >= 
#                                   self.similarity_threshold]
                
#                 if similar_patterns:
#                     # Make prediction based on similar patterns
#                     predicted_direction = max(set(p['direction'] for p in similar_patterns),
#                                            key=lambda x: sum(1 for p in similar_patterns 
#                                                            if p['direction'] == x))
#                     self.train_predictions.append({
#                         'timestamp': train_data.index[idx],
#                         'actual': direction,
#                         'predicted': predicted_direction,
#                         'pattern_length': pattern_length
#                     })
                
#                 # Store the pattern
#                 self.patterns_memory.append({
#                     'pattern': normalized_pattern,
#                     'direction': direction,
#                     'length': pattern_length
#                 })
    
#     def test(self, test_data: pd.DataFrame):
#         """Test the model on test data."""
#         significant_moves = self.identify_significant_moves(test_data)
#         logger.info(f"Found {len(significant_moves)} significant moves in test data")
        
#         for idx, direction in significant_moves:
#             for pattern_length in range(self.min_pattern_length, 
#                                       self.max_pattern_length + 1):
#                 pattern = self.extract_pattern(test_data, idx, pattern_length)
#                 if pattern is None:
#                     continue
                
#                 normalized_pattern = self.normalize_pattern(pattern)
#                 similar_patterns = [p for p in self.patterns_memory 
#                                   if len(p['pattern']) == len(normalized_pattern) and
#                                   self.calculate_similarity(p['pattern'], 
#                                                          normalized_pattern) >= 
#                                   self.similarity_threshold]
                
#                 if similar_patterns:
#                     predicted_direction = max(set(p['direction'] for p in similar_patterns),
#                                            key=lambda x: sum(1 for p in similar_patterns 
#                                                            if p['direction'] == x))
#                     self.test_predictions.append({
#                         'timestamp': test_data.index[idx],
#                         'actual': direction,
#                         'predicted': predicted_direction,
#                         'pattern_length': pattern_length
#                     })
    
#     def get_metrics(self, mode: str = 'all') -> dict:
#         """Calculate and return performance metrics."""
#         metrics = {}
        
#         if mode in ['train', 'all']:
#             train_total = len(self.train_predictions)
#             if train_total > 0:
#                 train_correct = sum(1 for p in self.train_predictions 
#                                   if p['actual'] == p['predicted'])
#                 metrics['train'] = {
#                     'total_predictions': train_total,
#                     'accuracy': train_correct / train_total,
#                     'win_rate': train_correct / train_total
#                 }
#             else:
#                 metrics['train'] = {'total_predictions': 0, 'accuracy': 0.0, 'win_rate': 0.0}
        
#         if mode in ['test', 'all']:
#             test_total = len(self.test_predictions)
#             if test_total > 0:
#                 test_correct = sum(1 for p in self.test_predictions 
#                                  if p['actual'] == p['predicted'])
#                 metrics['test'] = {
#                     'total_predictions': test_total,
#                     'accuracy': test_correct / test_total,
#                     'win_rate': test_correct / test_total
#                 }
#             else:
#                 metrics['test'] = {'total_predictions': 0, 'accuracy': 0.0, 'win_rate': 0.0}
        
#         metrics['total_patterns'] = len(self.patterns_memory)
#         return metrics

# def main():
#     # Initialize analyzer with training period
#     analyzer = MarketPatternAnalyzer(
#         min_pattern_length=4,
#         max_pattern_length=10,
#         movement_threshold=25.0,
#         similarity_threshold=0.85,
#         train_start='2021-01-01 09:15:00',
#         train_end='2025-01-01 09:15:00'
#     )
    
#     try:
#         # Load your data and split into train/test sets
#         train_data, test_data = analyzer.load_data('NIFTY_50_minute.csv')
        
#         # Train the model
#         logger.info("Training model...")
#         analyzer.train(train_data)
        
#         # Test the model
#         logger.info("Testing model...")
#         analyzer.test(test_data)
        
#         # Get and display metrics
#         metrics = analyzer.get_metrics(mode='all')
        
#         logger.info("\nTraining Results:")
#         logger.info(f"Total Patterns Stored: {metrics['total_patterns']}")
#         logger.info(f"Training Predictions: {metrics['train']['total_predictions']}")
#         logger.info(f"Training Accuracy: {metrics['train']['accuracy']:.2%}")
#         logger.info(f"Training Win Rate: {metrics['train']['win_rate']:.2%}")
        
#         logger.info("\nTesting Results:")
#         logger.info(f"Testing Predictions: {metrics['test']['total_predictions']}")
#         logger.info(f"Testing Accuracy: {metrics['test']['accuracy']:.2%}")
#         logger.info(f"Testing Win Rate: {metrics['test']['win_rate']:.2%}")
        
#     except Exception as e:
#         logger.error(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()

########################################################################

import pandas as pd
import numpy as np
from typing import List, Tuple
from datetime import datetime
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketPatternAnalyzer:
    def __init__(self, min_pattern_length: int = 4, max_pattern_length: int = 10, 
                 movement_threshold: float = 25.0, similarity_threshold: float = 0.85,
                 train_start: str = '2021-01-01 09:15:00',
                 train_end: str = '2025-01-01 09:15:00'):
        """
        Initialize the Market Pattern Analyzer.
        
        Args:
            min_pattern_length: Minimum number of candles to consider for pattern
            max_pattern_length: Maximum number of candles to consider for pattern
            movement_threshold: Minimum price movement to consider significant
            similarity_threshold: Threshold for pattern similarity (0-1)
            train_start: Start date for training period
            train_end: End date for training period
        """
        self.min_pattern_length = min_pattern_length
        self.max_pattern_length = max_pattern_length
        self.movement_threshold = movement_threshold
        self.similarity_threshold = similarity_threshold
        self.train_start = pd.to_datetime(train_start)
        self.train_end = pd.to_datetime(train_end)
        self.patterns_memory = []
        self.train_predictions = []
        self.test_predictions = []
        
    def load_data(self, file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and preprocess OHLC data from CSV file, split into train and test sets."""
        try:
            logger.info("Loading data...")
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Split data into training and testing sets
            train_data = df[self.train_start:self.train_end]
            test_data = df[self.train_end:]
            
            logger.info(f"Training data period: {train_data.index.min()} to {train_data.index.max()}")
            logger.info(f"Testing data period: {test_data.index.min()} to {test_data.index.max()}")
            
            return train_data, test_data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
            
    def normalize_pattern(self, pattern: pd.DataFrame) -> np.ndarray:
        """Normalize pattern for comparison."""
        closes = pattern['close'].values
        min_val = closes.min()
        max_val = closes.max()
        if max_val - min_val == 0:
            return np.zeros_like(closes)
        return (closes - min_val) / (max_val - min_val)
    
    def calculate_similarity(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
        """Calculate similarity between two patterns using correlation."""
        if len(pattern1) != len(pattern2):
            return 0
        correlation = np.corrcoef(pattern1, pattern2)[0, 1]
        return abs(correlation) if not np.isnan(correlation) else 0
    
    def identify_significant_moves(self, data: pd.DataFrame) -> List[Tuple[int, str]]:
        """Identify points where price moves more than threshold."""
        significant_moves = []
        for i in range(len(data) - 1):
            price_change = data['close'].iloc[i+1] - data['close'].iloc[i]
            if abs(price_change) >= self.movement_threshold:
                direction = 'up' if price_change > 0 else 'down'
                significant_moves.append((i, direction))
        return significant_moves
    
    def extract_pattern(self, data: pd.DataFrame, end_idx: int, 
                       pattern_length: int) -> pd.DataFrame:
        """Extract pattern of given length ending at specified index."""
        start_idx = end_idx - pattern_length + 1
        if start_idx < 0:
            return None
        return data.iloc[start_idx:end_idx + 1]
    
    def train(self, train_data: pd.DataFrame):
        """Train the model using training data."""
        significant_moves = self.identify_significant_moves(train_data)
        logger.info(f"Found {len(significant_moves)} significant moves in training data")
        
        # Create progress bar for training
        pbar = tqdm(significant_moves, desc="Training Progress", unit="pattern")
        
        for idx, direction in pbar:
            for pattern_length in range(self.min_pattern_length, 
                                      self.max_pattern_length + 1):
                pattern = self.extract_pattern(train_data, idx, pattern_length)
                if pattern is None:
                    continue
                
                normalized_pattern = self.normalize_pattern(pattern)
                similar_patterns = [p for p in self.patterns_memory 
                                  if len(p['pattern']) == len(normalized_pattern) and
                                  self.calculate_similarity(p['pattern'], 
                                                         normalized_pattern) >= 
                                  self.similarity_threshold]
                
                if similar_patterns:
                    predicted_direction = max(set(p['direction'] for p in similar_patterns),
                                           key=lambda x: sum(1 for p in similar_patterns 
                                                           if p['direction'] == x))
                    self.train_predictions.append({
                        'timestamp': train_data.index[idx],
                        'actual': direction,
                        'predicted': predicted_direction,
                        'pattern_length': pattern_length
                    })
                
                # Store the pattern
                self.patterns_memory.append({
                    'pattern': normalized_pattern,
                    'direction': direction,
                    'length': pattern_length
                })
                
            # Update progress bar description
            pbar.set_postfix({
                'Patterns': len(self.patterns_memory),
                'Predictions': len(self.train_predictions)
            })
    
    def test(self, test_data: pd.DataFrame):
        """Test the model on test data."""
        significant_moves = self.identify_significant_moves(test_data)
        logger.info(f"Found {len(significant_moves)} significant moves in test data")
        
        # Create progress bar for testing
        pbar = tqdm(significant_moves, desc="Testing Progress", unit="pattern")
        
        for idx, direction in pbar:
            for pattern_length in range(self.min_pattern_length, 
                                      self.max_pattern_length + 1):
                pattern = self.extract_pattern(test_data, idx, pattern_length)
                if pattern is None:
                    continue
                
                normalized_pattern = self.normalize_pattern(pattern)
                similar_patterns = [p for p in self.patterns_memory 
                                  if len(p['pattern']) == len(normalized_pattern) and
                                  self.calculate_similarity(p['pattern'], 
                                                         normalized_pattern) >= 
                                  self.similarity_threshold]
                
                if similar_patterns:
                    predicted_direction = max(set(p['direction'] for p in similar_patterns),
                                           key=lambda x: sum(1 for p in similar_patterns 
                                                           if p['direction'] == x))
                    self.test_predictions.append({
                        'timestamp': test_data.index[idx],
                        'actual': direction,
                        'predicted': predicted_direction,
                        'pattern_length': pattern_length
                    })
            
            # Update progress bar description
            pbar.set_postfix({
                'Predictions': len(self.test_predictions)
            })
    
    def get_metrics(self, mode: str = 'all') -> dict:
        """Calculate and return performance metrics."""
        metrics = {}
        
        if mode in ['train', 'all']:
            train_total = len(self.train_predictions)
            if train_total > 0:
                train_correct = sum(1 for p in self.train_predictions 
                                  if p['actual'] == p['predicted'])
                metrics['train'] = {
                    'total_predictions': train_total,
                    'accuracy': train_correct / train_total,
                    'win_rate': train_correct / train_total
                }
            else:
                metrics['train'] = {'total_predictions': 0, 'accuracy': 0.0, 'win_rate': 0.0}
        
        if mode in ['test', 'all']:
            test_total = len(self.test_predictions)
            if test_total > 0:
                test_correct = sum(1 for p in self.test_predictions 
                                 if p['actual'] == p['predicted'])
                metrics['test'] = {
                    'total_predictions': test_total,
                    'accuracy': test_correct / test_total,
                    'win_rate': test_correct / test_total
                }
            else:
                metrics['test'] = {'total_predictions': 0, 'accuracy': 0.0, 'win_rate': 0.0}
        
        metrics['total_patterns'] = len(self.patterns_memory)
        return metrics

def main():
    # Initialize analyzer with training period
    analyzer = MarketPatternAnalyzer(
        min_pattern_length=4,
        max_pattern_length=10,
        movement_threshold=25.0,
        similarity_threshold=0.85,
        train_start='2021-01-01 09:15:00',
        train_end='2025-01-01 09:15:00'
    )
    
    try:
        # Load your data and split into train/test sets
        train_data, test_data = analyzer.load_data('NIFTY_50_minute.csv')
        
        # Train the model
        logger.info("Starting training phase...")
        analyzer.train(train_data)
        
        # Test the model
        logger.info("\nStarting testing phase...")
        analyzer.test(test_data)
        
        # Get and display metrics
        metrics = analyzer.get_metrics(mode='all')
        
        logger.info("\nTraining Results:")
        logger.info(f"Total Patterns Stored: {metrics['total_patterns']}")
        logger.info(f"Training Predictions: {metrics['train']['total_predictions']}")
        logger.info(f"Training Accuracy: {metrics['train']['accuracy']:.2%}")
        logger.info(f"Training Win Rate: {metrics['train']['win_rate']:.2%}")
        
        logger.info("\nTesting Results:")
        logger.info(f"Testing Predictions: {metrics['test']['total_predictions']}")
        logger.info(f"Testing Accuracy: {metrics['test']['accuracy']:.2%}")
        logger.info(f"Testing Win Rate: {metrics['test']['win_rate']:.2%}")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()