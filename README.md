# AutoML-package
RANDOM STATE = 10

Klasyfkacja binarna

1. Preprocessing:
- Encoding (dla y też)
- Imputer (NAs)
- Scaling
- Feature selection - Golden features??

2. Selekcja i optymalizacja modeli:
- Random search?
- Można zakresy parametrów z tunability paper
- Kilka modeli + ensemble - stacking/voting (z wszystkich lub z kilku najlepszych)
- Dla niezbalansowanych danych w train test split jakiś StratifiedKfold czy coś - sprawdzić
- Potrzebujemy dużą precision bo lepiej powiedzieć że grzyb jest niejadalny (1- jadalny)

3. Ewaluacja i podsumowanie wyników:
- wszystko złączyć w klase na samym końcu i tam ma byc finalna wersja raportu 
+ nazwa: AutoML dla grzybiarzy 
- Można w 2 wybierać na podstawie custom_score i wtedy wyświetlić go i nazwe z parametrami i wyswietlic jaka
+ pie chart - balans klas ile z jednej ile z drugiej
- training time
- hiperparameter tunning: Random Search 
- AutoGluon will save models to "/Users/polamoscicka/Downloads/AutogluonModels/ag-20241211_175315"
Train Data Rows:    1548
- AutoML will use algorithms: ['Baseline', 'Decision Tree', 'Random Forest', 'Xgboost', 'Neural Network']
+ Train Data Columns: 18 -= ile categorical, ile missing
+ data imbalance
+ jaki preprocessing - top features - feature importance do selectfrommodel() - uzgodnić z Maćkiem - na treningowym??
+ Label Column:       label 
AutoML will ensemble available models
+ Roc auc plot
+ confusion matrix
- Funkcje które generują wykresy i pomysły na wykresy (bar plot wyników miary z kroswalidacji, żeby oceniać stabilność ??? (na koniec), + bar ploty jakichś najważniejszych miar)
- dodac jakies korelacje miar między algorytmami - żeby zobaczyć które modele mają podobne wyniki różnych miar - jak użytkownikowi będzie zależało najbardziej np na precyzji to bedzie mozna tego modelu użyć? ?????
- cos z tym balansowanuem
-
-  model comparison miay - opcjonalne
+ Wnioski z accuracy, sensivity, recall I precission połączone ze zbalansowaniem zbioru np

# Datasets 1,2,4
https://www.kaggle.com/datasets/prishasawhney/mushroom-dataset
https://www.kaggle.com/datasets/devzohaib/mushroom-edibility-classification
https://www.kaggle.com/datasets/ulrikthygepedersen/mushroom-attributes
https://www.kaggle.com/datasets/uciml/mushroom-classification
https://www.kaggle.com/datasets/bwandowando/mushroom-overload
https://www.kaggle.com/datasets/vishalpnaik/mushroom-classification-edible-or-poisonous

# Other implementations:
https://github.com/awesomecosmos/Mushroom-Classification
https://www.researchgate.net/publication/369422963_Predicting_Mushroom_Edibility_with_Effective_Classification_and_Efficient_Feature_Selection_Techniques
