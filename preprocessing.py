"""
Data Preprocessing Module for Bank Marketing Dataset

This module provides reusable and reproducible data preprocessing functions
for preparing the bank marketing dataset for model training.

Author: ML Engineering Team
Date: November 2025
"""

import pandas as pd
import numpy as np
import yaml
import pickle
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Any
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BankMarketingPreprocessor:
    """
    Preprocessor for Bank Marketing dataset.
    
    This class handles all preprocessing steps including:
    - Feature engineering
    - Outlier removal
    - Categorical encoding
    - Train-test splitting
    
    All transformations are fitted on training data and can be applied to new data.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.label_encoders = {}
        self.feature_names = None
        self.fitted = False
        
        # Set random seed for reproducibility
        np.random.seed(self.config['general']['random_seed'])
        
        logger.info("Preprocessor initialized with config from %s", config_path)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Fit the preprocessor on training data and transform it.
        
        Args:
            df: Raw dataframe with all features and target
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Starting fit_transform on dataset with shape %s", df.shape)
        
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # 1. Feature Engineering
        df = self._feature_engineering(df)
        logger.info("Feature engineering completed. New shape: %s", df.shape)
        
        # 2. Outlier Removal
        df = self._remove_outliers(df)
        logger.info("Outlier removal completed. New shape: %s", df.shape)
        
        # 3. Separate features and target
        X, y = self._separate_features_target(df)
        
        # 4. Encode target variable
        y = self._encode_target(y)
        
        # 5. Train-test split (before encoding to prevent data leakage)
        X_train, X_test, y_train, y_test = self._train_test_split(X, y)
        logger.info("Train-test split completed. Train shape: %s, Test shape: %s", 
                   X_train.shape, X_test.shape)
        
        # 6. Categorical Encoding (fit on train, transform both)
        X_train = self._fit_categorical_encoding(X_train)
        X_test = self._transform_categorical_encoding(X_test)
        logger.info("Categorical encoding completed")
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        self.fitted = True
        
        logger.info("Preprocessing completed successfully")
        return X_train, X_test, y_train, y_test
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessor.
        
        Args:
            df: Raw dataframe with all features (no target)
            
        Returns:
            Preprocessed dataframe ready for prediction
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform. Call fit_transform first.")
        
        logger.info("Transforming new data with shape %s", df.shape)
        
        # Create a copy
        df = df.copy()
        
        # Apply same transformations (without outlier removal and train-test split)
        df = self._feature_engineering(df)
        X = self._transform_categorical_encoding(df)
        
        # Ensure columns match training data
        X = X.reindex(columns=self.feature_names, fill_value=0)
        
        logger.info("Transform completed. Output shape: %s", X.shape)
        return X
    
    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering transformations.
        
        - Remove 'day' feature
        - Split 'pdays' into 'was_contacted_before' and 'days_since_contact'
        """
        df = df.copy()
        config = self.config['feature_engineering']
        
        # Remove specified features
        for feature in config['features_to_drop']:
            if feature in df.columns:
                df = df.drop(columns=[feature])
                logger.debug("Dropped feature: %s", feature)
        
        # Transform pdays if it exists
        if 'pdays' in df.columns:
            pdays_config = config['pdays']
            never_contacted = pdays_config['never_contacted_value']
            
            # Create binary feature: was_contacted_before
            if pdays_config['create_binary_feature']:
                df['was_contacted_before'] = (df['pdays'] != never_contacted).astype(int)
                # Convert to 'yes'/'no' for consistency with other binary features
                df['was_contacted_before'] = df['was_contacted_before'].map({1: 'yes', 0: 'no'})
            
            # Create numeric feature: days_since_contact
            if pdays_config['create_days_feature']:
                df['days_since_contact'] = df['pdays'].copy()
                df.loc[df['pdays'] == never_contacted, 'days_since_contact'] = pdays_config['days_for_never_contacted']
            
            # Drop original pdays
            df = df.drop(columns=['pdays'])
            logger.debug("Transformed pdays into was_contacted_before and days_since_contact")
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove outliers based on IQR method and specific thresholds.
        
        - Apply 3×IQR rule for: age, balance, duration, campaign
        - Remove rows with previous > 50
        - Remove rows with days_since_contact > 800
        """
        df = df.copy()
        config = self.config['outlier_removal']
        initial_shape = df.shape[0]
        
        # Apply IQR method
        iqr_features = config['iqr_features']
        iqr_multiplier = config['iqr_multiplier']
        
        for feature in iqr_features:
            if feature in df.columns:
                Q1 = df[feature].quantile(0.25)
                Q3 = df[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - iqr_multiplier * IQR
                upper_bound = Q3 + iqr_multiplier * IQR
                
                outliers_count = ((df[feature] < lower_bound) | (df[feature] > upper_bound)).sum()
                df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
                logger.debug("Removed %d outliers from %s using IQR method", outliers_count, feature)
        
        # Apply specific threshold removals
        thresholds = config['threshold_removals']
        
        for feature, params in thresholds.items():
            if feature in df.columns:
                if 'max_value' in params:
                    outliers_count = (df[feature] > params['max_value']).sum()
                    df = df[df[feature] <= params['max_value']]
                    logger.debug("Removed %d rows where %s > %d", 
                               outliers_count, feature, params['max_value'])
        
        rows_removed = initial_shape - df.shape[0]
        logger.info("Outlier removal: removed %d rows (%.2f%%)", 
                   rows_removed, (rows_removed / initial_shape) * 100)
        
        return df
    
    def _separate_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Separate features and target variable."""
        target_col = self.config['data']['target_column']
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")
        
        y = df[target_col]
        X = df.drop(columns=[target_col])
        
        return X, y
    
    def _encode_target(self, y: pd.Series) -> pd.Series:
        """
        Encode target variable from 'yes'/'no' to 1/0.
        
        Args:
            y: Target series with 'yes'/'no' values
            
        Returns:
            Binary encoded target (1 for 'yes', 0 for 'no')
        """
        target_mapping = {'yes': 1, 'no': 0}
        y_encoded = y.map(target_mapping)
        
        if y_encoded.isnull().any():
            raise ValueError("Target variable contains values other than 'yes' and 'no'")
        
        logger.debug("Target variable encoded: yes=1, no=0")
        return y_encoded
    
    def _train_test_split(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        """
        Split data into training and test sets with stratification.
        
        Args:
            X: Features dataframe
            y: Target series
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        data_config = self.config['data']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=data_config['test_size'],
            stratify=y if data_config['stratify'] else None,
            random_state=self.config['general']['random_seed']
        )
        
        return X_train, X_test, y_train, y_test
    
    def _fit_categorical_encoding(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform categorical features using label encoding.
        
        This method fits the encoders on the training data.
        
        Args:
            X: Training features dataframe
            
        Returns:
            Encoded dataframe
        """
        X = X.copy()
        config = self.config['categorical_encoding']
        categorical_features = config['categorical_features']
        
        for feature in categorical_features:
            if feature in X.columns:
                # Initialize and fit encoder
                le = LabelEncoder()
                X[feature] = le.fit_transform(X[feature].astype(str))
                
                # Store encoder for later use
                self.label_encoders[feature] = le
                logger.debug("Fitted label encoder for %s: %d classes", 
                           feature, len(le.classes_))
        
        return X
    
    def _transform_categorical_encoding(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform categorical features using fitted encoders.
        
        This method transforms new data using encoders fitted on training data.
        Handles unseen categories by assigning them to a default value.
        
        Args:
            X: Features dataframe to encode
            
        Returns:
            Encoded dataframe
        """
        X = X.copy()
        
        for feature, le in self.label_encoders.items():
            if feature in X.columns:
                # Handle unseen categories
                X[feature] = X[feature].astype(str)
                
                # Map known categories, unknown ones get -1
                known_categories = set(le.classes_)
                X[feature] = X[feature].apply(
                    lambda x: le.transform([x])[0] if x in known_categories else -1
                )
                
                logger.debug("Transformed %s using fitted encoder", feature)
        
        return X
    
    def save(self, filepath: str) -> None:
        """
        Save the fitted preprocessor to disk.
        
        Args:
            filepath: Path where to save the preprocessor
        """
        if not self.fitted:
            raise ValueError("Cannot save unfitted preprocessor")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        preprocessor_state = {
            'config': self.config,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'fitted': self.fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessor_state, f)
        
        logger.info("Preprocessor saved to %s", filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'BankMarketingPreprocessor':
        """
        Load a fitted preprocessor from disk.
        
        Args:
            filepath: Path to the saved preprocessor
            
        Returns:
            Loaded preprocessor instance
        """
        with open(filepath, 'rb') as f:
            preprocessor_state = pickle.load(f)
        
        # Create instance with loaded config
        instance = cls.__new__(cls)
        instance.config = preprocessor_state['config']
        instance.label_encoders = preprocessor_state['label_encoders']
        instance.feature_names = preprocessor_state['feature_names']
        instance.fitted = preprocessor_state['fitted']
        
        logger.info("Preprocessor loaded from %s", filepath)
        return instance


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load bank marketing dataset from CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with loaded data
    """
    logger.info("Loading data from %s", filepath)
    
    # Try different delimiters common in this dataset
    try:
        df = pd.read_csv(filepath, sep=';')
    except:
        df = pd.read_csv(filepath)
    
    logger.info("Data loaded successfully. Shape: %s", df.shape)
    logger.info("Columns: %s", list(df.columns))
    
    return df


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("Bank Marketing Data Preprocessor")
    print("=" * 80)
    
    # This is an example - you'll need to provide the actual data file
    # df = load_data("path/to/bank-marketing.csv")
    # preprocessor = BankMarketingPreprocessor("config.yaml")
    # X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
    
    print("\nPreprocessor module loaded successfully!")
    print("\nTo use:")
    print("1. Load your data: df = load_data('path/to/data.csv')")
    print("2. Create preprocessor: preprocessor = BankMarketingPreprocessor('config.yaml')")
    print("3. Fit and transform: X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)")
    print("4. Save preprocessor: preprocessor.save('preprocessor.pkl')")
