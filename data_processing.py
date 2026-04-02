import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path):
    """Load the dataset and separate features and target."""
    logger.info(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path, sep=';')
        if len(df.columns) <= 1:
             df = pd.read_csv(file_path, sep=',')
    except Exception as e:
        logger.warning(f"Failed to load with standard bank logic, defaulting to comma: {e}")
        df = pd.read_csv(file_path, sep=',')
        
    if 'y' not in df.columns:
        raise ValueError("Target column 'y' is missing from the dataset.")

    # Encode target variable
    df['y'] = df['y'].map({'yes': 1, 'no': 0})
    
    X = df.drop('y', axis=1)
    y = df['y']
    return X, y

def get_preprocessor(X):
    """Builds a scikit-learn ColumnTransformer for preprocessing numerical and categorical features."""
    logger.info("Building preprocessing pipeline...")
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor
