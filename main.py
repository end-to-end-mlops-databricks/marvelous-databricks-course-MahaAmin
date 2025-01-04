import yaml
import logging

from fraud_credit_cards.data_processor import DataProcessor
from fraud_credit_cards.fraud_model import FraudModel
from colorama import Back, Fore, Style
from sklearn.metrics import accuracy_score, classification_report

def print_evaluation(y_test, y_pred, accuracy):
    print("Accuracy:", accuracy)
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

# configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# load configurations
with open('project_config.yml', 'r') as file:
    config = yaml.safe_load(file)

logger.info("Configuration loaded: ")
print(yaml.dump(config, default_flow_style=False))

# initialize DataProcessor
data_processor = DataProcessor("data/creditcard_2023.csv", config)
logging.info("DataProcessor Initialized ...")

# preprocess the data
data_processor.preprocess_data()
logging.info("Data preprocessed ...")

# Split the data
X_train, X_test, y_train, y_test = data_processor.split_data()
logger.info(f"Data split into training and test sets.")
logger.debug(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# Intialize and train model
model = FraudModel(data_processor.preprocessor)
model.train(X_train,y_train)
logger.info("Model training completed.")

# evaluate model
y_pred, accuracy, precision, recall, mse, f1 = model.evaluate(X_test, y_test)
logging.info("Model evaluation completed. ")

# print evaluation report
print_evaluation(y_test, y_pred, accuracy)
