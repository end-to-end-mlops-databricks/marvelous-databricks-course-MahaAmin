from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, r2_score


class FraudModel:
    def __init__(self, preprocessor):
        self.model = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier', CatBoostClassifier(verbose=False))])
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        # Predict on the test set
        y_pred = self.predict(X_test)

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

        return accuracy, precision, recall, mse, f1
