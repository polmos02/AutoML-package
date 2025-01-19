# AutoML-package
RANDOM STATE = 10

Klasyfkacja binarna

1. Preprocessing:
- Encoding
- Imputer
- Scaling
- Feature selection

2. Selekcja i optymalizacja modeli:
- Random search
- Można zakresy parametrów z tunability paper
- Kilka modeli + ensemble - stacking/voting (z wszystkich lub z kilku najlepszych)
- Dla niezbalansowanych danych w train test split jakiś StratifiedKfold czy coś - sprawdzić
- Potrzebujemy dużą precision bo lepiej powiedzieć że grzyb jest niejadalny (1- jadalny)

3. Ewaluacja i podsumowanie wyników:


# Datasets
https://www.kaggle.com/datasets/prishasawhney/mushroom-dataset
https://www.kaggle.com/datasets/devzohaib/mushroom-edibility-classification
https://www.kaggle.com/datasets/uciml/mushroom-classification

# Other implementations:
https://github.com/awesomecosmos/Mushroom-Classification
https://www.researchgate.net/publication/369422963_Predicting_Mushroom_Edibility_with_Effective_Classification_and_Efficient_Feature_Selection_Techniques
