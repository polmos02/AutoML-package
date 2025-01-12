from pakietAutoML.automl_functions import *
from pakietAutoML.modul1 import *
from pakietAutoML.model_selection import *
from pakietAutoML.load_models import *

def generate_raport(X_train, y_train, X_test, y_test, X_train_or, y_train_or):
    model,best_score =model_selection(X_train,y_train)
    y_pred=model.predict(X_test)
    metrics=model_evaluation(y_pred, y_test)
    weighted_score=custom_score(y_test,y_pred)
    print("Pakiet AutoML dla grzybiarzy")
    print("Analizowane są zbiory danych z podziałem na klasy 0 lub 1, gdzie 0 oznacza jadalny gdzyb, a 1 trujący.")
    print("Analiza danych:")
    data_overview(X_train_or)
    print("Balans klas:")
    plot_mushroom_balance(y_train_or)
    print("Preprocessing: TODO")
    print("Ważność cech:")
    _, selected_features = feature_select(X_train, y_train)
    summarize_selected_features(selected_features)
    
    model.fit(X_train,y_train)

    print("Analiza jakości modeli i konfiguracja finalnego komitetu:")
    print("1. Miara oceny modeli:")
    print("   Do analizy jakości modeli wykorzystano kombinację ważonych miar ROC AUC oraz Recall:")
    print("   Custom Score = (Recall: 0.3, ROC AUC: 0.7)")
    print()
    print("2. Modele użyte w analizie: KNeighborsClassifier, GradientBoostingClassifier, RandomForestClassifier, LogisticRegression")
    print()
    print("3. Optymalizacja parametrów:")
    print("   Dla każdego z modeli, przy użyciu metody RandomizedSearch, dobrano najlepsze zestawy hiperparametrów.")
    print()
    print("4. Finalny komitet modeli:")
    print("   Modele z optymalnymi parametrami zostały połączone w komitet VotingClassifier, reprezentujący finalny model.")
    print()
    print("5. Parametry finalnego modelu:")
    print(model)
    print()
    print("6. Wynik Custom Score:")
    print("   Uzyskana wartość Custom Score dla tego modelu na zbiorze walidacyjnym wynosiła:")
    print(weighted_score)
    # Plot Confusion Matrix
    plot_confusion_matrix(metrics['confusion_matrix'])
    # Plot ROC AUC Curve
    plot_roc_auc_curve(metrics['roc_curve'][0], metrics['roc_curve'][1], metrics['roc_auc'])
    # Plot Bar Plot of Metrics
    plot_metrics_bar(metrics, weighted_score)

    generate_model_analysis_from_metrics(metrics)






