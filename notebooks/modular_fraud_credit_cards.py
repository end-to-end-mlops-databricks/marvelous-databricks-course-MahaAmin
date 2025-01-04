# Databricks notebook source
# MAGIC %md
# MAGIC # Credit Card 2023 Fraud Detection
# MAGIC
# MAGIC **Dataset:** [Kaggle-Credit-Card-Fraud-Detection-Dataset-2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023/data)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Installing Packages

# COMMAND ----------

# MAGIC %pip install colorama==0.4.6 catboost==1.2.0 gecs==0.1.1

# COMMAND ----------

# MAGIC %md
# MAGIC ### Import Libraries

# COMMAND ----------


import warnings

import pandas as pd
from catboost import CatBoostClassifier
from colorama import Fore, Style
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reading Data From DB Catalog Volume

# COMMAND ----------


def load_data(filepath):
    """
    Load the data from the given filepath.
    """
    df = pd.read_csv(filepath)
    return df


filepath = "/Volumes/fraud_credit_cards/data/credit_cards_2023/creditcard_2023.csv"
df = load_data(filepath)
df.head()


# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Preprocessing

# COMMAND ----------


def preprocess_data(df, target_column="Class"):
    # Spliting the data into features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Define numeric features (remove categorical columns)
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Define preprocessing steps
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, numeric_features)])

    print("Features Shape: ", X.shape)
    print("Target Shape: ", y.shape)
    return X, y, preprocessor


X, y, preprocessor = preprocess_data(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Modeling and Evaluation

# COMMAND ----------


def train_and_evaluate_model(X, y, preprocessor, test_size=0.2, random_state=42, n_estimators=100):
    # Define the model
    model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", CatBoostClassifier(verbose=False))])

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Fit the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    return model, X_train, X_test, y_train, y_test, y_pred


model, X_train, X_test, y_train, y_test, y_pred = train_and_evaluate_model(X, y, preprocessor)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Evaluation

# COMMAND ----------


def evaluate_model(model, X_train, X_test, y_train, y_test, y_pred):
    # Generate evaluation metrics
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Calculate precision
    precision = precision_score(y_test, y_pred)

    # Calculate recall
    recall = recall_score(y_test, y_pred)

    # Calculate Mean-Squared Error
    mse = mean_squared_error(y_test, y_pred)

    # Calcualte F1 score
    f1 = f1_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("MSE:", mse)
    print("F1 Score:", f1)

    # Display classification report with colors and heading
    from colorama import Back

    print("\n" + Back.BLUE + Fore.WHITE + "Classification Report" + Style.RESET_ALL)
    report = classification_report(y_test, y_pred, output_dict=True)
    for key, value in report.items():
        if key in ["0", "1"]:
            color = Fore.GREEN if value["precision"] > 0.8 else Fore.RED
            print(f"Class {key}:")
            print(f"  Precision: {color}{value['precision']:.2f}{Style.RESET_ALL}")
            color = Fore.GREEN if value["recall"] > 0.8 else Fore.RED
            print(f"  Recall: {color}{value['recall']:.2f}{Style.RESET_ALL}")
            color = Fore.GREEN if value["f1-score"] > 0.8 else Fore.RED
            print(f"  F1-score: {color}{value['f1-score']:.2f}{Style.RESET_ALL}")
            print(f"  Support: {value['support']}")
        else:
            print(key + ":", value)
    return accuracy, precision, recall, mse, f1


accuracy, precision, recall, mse, f1 = evaluate_model(model, X_train, X_test, y_train, y_test, y_pred)


# COMMAND ----------
