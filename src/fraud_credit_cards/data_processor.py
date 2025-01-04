import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class DataProcessor:
    def __init__(self, file_path, config):
        self.df = self.load_data(file_path)
        self.config = config
        self.X = None
        self.y = None
        self.preprocessor = None

    def load_data(self, file_path):
        """
        Load the data from the given filepath.
        """
        df = pd.read_csv(file_path)
        return df

    def preprocess_data(self):
        # Spliting the data into features and target
        target = self.config["target"]
        self.X = self.df.drop(target, axis=1)
        self.y = self.df[target]

        # Define numeric features (remove categorical columns)
        numeric_features = self.X.select_dtypes(include=["int64", "float64"]).columns.tolist()

        # Define preprocessing steps
        numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

        self.preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, numeric_features)])

    # split dataset
    def split_data(self, test_size=0.2, random_state=42):
        # Split the data into training and test sets
        return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
