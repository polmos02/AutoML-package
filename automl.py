from sklearn.model_selection import train_test_split
import time

from modul1 import prep
from model_selection import model_selection, model_evaluation

class AutoML:

    def __init__(self):
        self.best_model = None
        self.best_score = None
        self.selected_features = None
        self.X = None
        self.y = None
        self.metrics = None
        self.fit_time = None

    def get_selected_features(self):
        if not self.selected_features:
            raise Exception("Model is not trained yet. Call fit() before get_selected_features().")
        return self.selected_features

    def get_best_model(self):
        if not self.best_model:
            raise Exception("Model is not trained yet. Call fit() before get_best_model().")
        return self.best_model

    def get_best_score(self):
        if not self.best_score:
            raise Exception("Model is not trained yet. Call fit() before get_best_score().")
        return self.best_score

    def get_metrics(self):
        if not self.metrics:
            raise Exception("Model is not trained yet. Call fit() before get_metrics().")

        return self.metrics

    def get_fit_time(self):
        if not self.fit_time:
            raise Exception("Model is not trained yet. Call fit() before get_fit_time().")

        return self.fit_time

    def fit(self, X, y):
        TEST_SIZE = 0.2
        RANDOM_STATE = 10

        try:
            start_time = time.time()

            self.X = X
            self.y = y

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

            # preprocesing
            X_train_preprocessed, self.selected_features = prep(X_train, y_train, mode='train')
            X_test_preprocessed = prep(X_test, mode='test', features=self.selected_features)

            # model selection
            self.best_model, self.best_score = model_selection(X_train_preprocessed, y_train)
            if self.best_model is None:
                raise ValueError("Model selection failed. No model was returned.")

            # evaluate on test set
            y_pred = self.best_model.predict(X_test_preprocessed)
            self.metrics = model_evaluation(y_test, y_pred)

            self.fit_time = time.time() - start_time


        except Exception as e:
            print(f"An error occurred during the AutoML process: {str(e)}")
            raise

    def predict(self, X):

        if not self.best_model:
            raise Exception("Model is not trained yet. Call fit() before predict().")

        # preprocess the input data
        X_preprocessed = prep(X, mode='test', features=self.selected_features)

        # predict
        return self.best_model.predict(X_preprocessed)

    def summary_report(self):
        # TODO: Funkcja Poli
        pass


# Example usage
# if __name__ == "__main__":
#
#     automl = AutoML()
#     import pandas as pd
#     from sklearn.datasets import load_breast_cancer
#     data = load_breast_cancer()
#     X = pd.DataFrame(data.data, columns=data.feature_names)
#     y = pd.Series(data.target, name="target")
#
#     automl.fit(X, y)
#
#     y_pred = automl.predict(X)
#     print(f"Predictions: {y_pred}")
#
#     metrics = automl.get_metrics()
#     print(f"Metrics: {metrics}")
#
#     fit_time = automl.get_fit_time()
#     print(f"Fit time: {fit_time} seconds")
#
#     print(automl.get_best_model())
#     print(automl.get_best_score())

