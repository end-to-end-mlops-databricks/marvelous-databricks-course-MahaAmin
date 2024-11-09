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

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
from IPython.display import display, HTML
import warnings
from colorama import Fore, Style

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import randint
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings('ignore')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reading Data From DB Catalog Volume

# COMMAND ----------

df_tr = pd.read_csv('/Volumes/fraud_credit_cards/data/credit_cards_2023/creditcard_2023.csv')


# COMMAND ----------

df_tr.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Preprocessing and Modeling

# COMMAND ----------

# Spliting the data into features and target
X = df_tr.drop('Class', axis=1)
y = df_tr['Class']

# Define numeric features (remove categorical columns)
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Define preprocessing steps
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)])

# Define the model
model = Pipeline(steps=[('preprocessor', preprocessor),
                         ('classifier', CatBoostClassifier(verbose=False))])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Display classification report with colors and heading
from colorama import Fore, Back, Style
print("\n" + Back.BLUE + Fore.WHITE + "Classification Report" + Style.RESET_ALL)
report = classification_report(y_test, y_pred, output_dict=True)
for key, value in report.items():
    if key in ['0', '1']:  
        color = Fore.GREEN if value['precision'] > 0.8 else Fore.RED
        print(f"Class {key}:")
        print(f"  Precision: {color}{value['precision']:.2f}{Style.RESET_ALL}")
        color = Fore.GREEN if value['recall'] > 0.8 else Fore.RED
        print(f"  Recall: {color}{value['recall']:.2f}{Style.RESET_ALL}")
        color = Fore.GREEN if value['f1-score'] > 0.8 else Fore.RED
        print(f"  F1-score: {color}{value['f1-score']:.2f}{Style.RESET_ALL}")
        print(f"  Support: {value['support']}")
    else:
        print(key + ":", value)

# COMMAND ----------


