from sklearn.datasets import load_iris
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
from sklearn.metrics import accuracy_score

# Load Data
iris  = load_iris()

# Split train, test 
x_train, x_test, y_train, y_test = train_test_split(
    iris.data, 
    iris.target, 
    test_size = 0.2, 
    random_state = 42
)

# Feature Engineering -> Preprocessing -> Standardization 
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)
# -------------
estimator = KNeighborsClassifier()

# Create a list of param_dict for grid search 
param_dict = {'n_neighbors': [i for i in range(1, 11)]} 

# Create GridSearchCV object
# Every grid search will be calculated four folds, so 40 times
estimator = GridSearchCV(estimator, param_dict, cv = 4)

# Model Training
estimator.fit(x_train, y_train)

print(estimator.best_score_)
print('----')
print(estimator.best_estimator_)
print('----')
print(estimator.best_params_)
print('----')
print(estimator.cv_results_)

# Model Eval
# Once GridSearchCV finishes cross-validation, it automatically stores the best parameters, the best model, and the best score — you don’t need to manually input k or retrain the model yourself.
y_pred = estimator.predict(x_test)
score = accuracy_score(y_test, y_pred)
print(f'the accuracy score is {score}')
