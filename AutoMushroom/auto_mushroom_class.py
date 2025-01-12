from sklearn.model_selection import train_test_split
import time

from AutoMushroom.preprocessing import prep
from AutoMushroom.model_selection import *
from AutoMushroom.report import *

import warnings


class AutoMushroom:
    warnings.filterwarnings("ignore", category=FutureWarning)

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

    def fit(self, X, y, mode = 'medium'):
        TEST_SIZE = 0.1
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
            self.best_model, self.best_score = model_selection(X_train_preprocessed, y_train, mode = mode)
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

    # def predict_proba(self, X):
    #
    #         if not self.best_model:
    #             raise Exception("Model is not trained yet. Call fit() before predict().")
    #
    #         # preprocess the input data
    #         X_preprocessed = prep(X, mode='test', features=self.selected_features)
    #
    #         # predict
    #         return self.best_model.predict_proba(X_preprocessed)


    def summary_report(self):

        if not self.best_model:
            raise Exception("Model is not trained yet. Call fit() before summary_report().")

        print("Pakiet AutoML dla grzybiarzy")
        print("Analizowane są zbiory danych z podziałem na klasy 0 lub 1, gdzie 0 oznacza jadalny grzyb, a 1 trujący.")
        print("Analiza danych:")
        data_overview(self.X)
        print("Balans klas:")
        plot_mushroom_balance(self.y)
        print("Preprocessing składa się z kilku etapów:")
        print("Numeryczne dane są wypełniane średnią w przypadku braków, a następnie skalowane do zakresu [0,1] przy użyciu MinMaxScaler.")
        print("Dane kategoryczne są uzupełniane najczęściej występującymi wartościami, a następnie kodowane za pomocą metody one-hot encoding.")
        print("W trybie treningowym wybierane są istotne cechy za pomocą klasyfikatora Random Forest i SelectFromModel, a dane testowe są ograniczane do wybranych cech.")
        print("Ważność cech:")
        summarize_selected_features(self.selected_features)

        print("Analiza jakości modeli i konfiguracja finalnego komitetu:")
        print("1. Miara oceny modeli:")
        print("   Do analizy jakości modeli wykorzystano kombinację ważonych miar ROC AUC oraz Recall:")
        print("   Custom Score = (Recall: 0.3, ROC AUC: 0.7)")
        print()
        print(
            "2. Modele użyte w analizie: KNeighborsClassifier, GradientBoostingClassifier, RandomForestClassifier, LogisticRegression")
        print(
            "   Dodatkowo komitet VotingClassifier z wyżej wymienionych modeli z optymalnymi parametrami")
        print()
        print("3. Optymalizacja parametrów:")
        print(
            "   Dla każdego z modeli, przy użyciu metody RandomizedSearch, dobrano najlepsze zestawy hiperparametrów.")
        print()
        print("4. Parametry finalnego modelu:")
        return self.best_model
        print()
        print(f"5. Czas trenowania modelu: {self.fit_time} seconds")
        print()
        print("6. Wynik Custom Score:")
        print(f"   Uzyskana wartość Custom Score dla tego modelu na zbiorze walidacyjnym wynosiła: {self.best_score}")
        # Plot Confusion Matrix
        plot_confusion_matrix(self.metrics['confusion_matrix'])
        # Plot ROC AUC Curve
        plot_roc_auc_curve(self.metrics['roc_curve'][0], self.metrics['roc_curve'][1], self.metrics['roc_auc'])
        # Plot Bar Plot of Metrics
        plot_metrics_bar(self.metrics, self.best_score)

        generate_model_analysis_from_metrics(self.metrics)


# #Example usage
if __name__ == "__main__":

    automl = AutoML()
    import pandas as pd
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    automl.fit(X_train, y_train)

    y_pred = automl.predict(X)
    print(f"Predictions: {y_pred}")
    
    metrics = automl.get_metrics()
    print(f"Metrics: {metrics}")
    
    
    print(automl.get_best_model())
    print(automl.get_best_score())

    print(automl.predict(X_test))

