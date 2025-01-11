from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import VotingClassifier

import json
import importlib

def load_models():
    # load models from JSON file
    with open('models.json', 'r') as file:
        models_json = json.load(file)

    # create dictionary with models and parameters
    models = {}
    for model_info in models_json['models']:
        # import models
        module = importlib.import_module(model_info['module'])
        model_class = getattr(module, model_info['name'])

        models[model_info['name']] = {
            'model': model_class,
            'params': model_info['params']
        }

    return models

def model_evaluation(y_pred, y_test):

    # calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': confusion,
        'roc_curve': (fpr, tpr)
    }

    return metrics

def custom_score(y_test, y_pred):

    # custom score for random search
    weights = {"recall": 0.3, "roc_auc": 0.7}
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    weighted_score = weights["recall"] * recall + weights["roc_auc"] * roc_auc

    return weighted_score

def model_selection(X_train, y_train):
    CV = 5
    N_ITER = 5
    RANDOM_STATE = 10

    best_model = None
    best_score = 0
    models_list = []

    # load the models
    models_dict = load_models()

    for model_info in models_dict.values():

        # get the params from dict
        param_distributions = model_info['params']

        # custom scoring based on function
        custom_scorer = make_scorer(custom_score, greater_is_better=True)

        param_search = RandomizedSearchCV(
                                    model_info['model'](),
                                    param_distributions=param_distributions,
                                    n_iter=N_ITER,
                                    cv=CV,
                                    random_state=RANDOM_STATE,
                                    scoring = custom_scorer,
                                    n_jobs=-1
        )

        param_search.fit(X_train, y_train)

        rs_best_model = param_search.best_estimator_
        score = param_search.best_score_

        # Update the best model
        if score > best_score:
            best_model = rs_best_model
            best_score = score

        models_list.append((model_info['model'].__name__, rs_best_model))

    ensemble_model = VotingClassifier(estimators=models_list, voting='soft', n_jobs=-1)
    ensemble_model.fit(X_train, y_train)

    ensemble_score = custom_score(y_train, ensemble_model.predict(X_train))

    if ensemble_score > best_score:
        best_model = ensemble_model
        best_score = ensemble_score

    return best_model, best_score


