### Iris Dataset

##### Steps: 
1. Import packages
2. Load data + EDA
3. Data preprocessing
4. Feature engineering 
5. Model training
6. Model prediction 
7. Model evaluation

-- import pacakges: 
```python
from sklearn.datasets import load_iris
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
```
-- Load data
```python
iris = load_iris()
iris_data = load_iris(as_frame = True)
df = pd.DataFrame(iris_data.frame)
```
-- Explortary Data Analysis EDA
```python
def eda():
    print('EDA starts: ')
    print(iris.keys())
    print('---')
    print(df.head()) 
    print('---')
    print(df.groupby('target').size())
    print('---')
    for key, target in enumerate(iris.target_names):
        print(f'{target}: {key}')
    print('---, descriptive statistics below')
    print(iris.DESCR)

"""
EDA starts: 
dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
---
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target
0                5.1               3.5                1.4               0.2       0
1                4.9               3.0                1.4               0.2       0
2                4.7               3.2                1.3               0.2       0
3                4.6               3.1                1.5               0.2       0
4                5.0               3.6                1.4               0.2       0
---
target
0    50
1    50
2    50
dtype: int64
---
setosa: 0
versicolor: 1
virginica: 2
---, descriptive statistics below
.. _iris_dataset:

Iris plants dataset
--------------------

**Data Set Characteristics:**

:Number of Instances: 150 (50 in each of three classes)
:Number of Attributes: 4 numeric, predictive attributes and the class
:Attribute Information:
    - sepal length in cm
    - sepal width in cm
    - petal length in cm
    - petal width in cm
    - class:
            - Iris-Setosa
            - Iris-Versicolour
            - Iris-Virginica

:Summary Statistics:

============== ==== ==== ======= ===== ====================
                Min  Max   Mean    SD   Class Correlation
============== ==== ==== ======= ===== ====================
sepal length:   4.3  7.9   5.84   0.83    0.7826
sepal width:    2.0  4.4   3.05   0.43   -0.4194
petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
petal width:    0.1  2.5   1.20   0.76    0.9565  (high!)
============== ==== ==== ======= ===== ====================

:Missing Attribute Values: None
:Class Distribution: 33.3% for each of 3 classes.
:Creator: R.A. Fisher
:Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
:Date: July, 1988
"""
```
-- Data Visualization 
```python
def visualization():
    # generating a pairplot to display the relationship between two features pairwisely
    sns.pairplot(df, hue="target", diag_kind="kde", corner=True)
    plt.show()
```
<p align="center">
<img src= 'assets/pairplot.png'
width=""/>
</p>

-- Preprocessig + Feature Engineering + Model Traing + Model Eval

```python
def model():
    # Data preprocessing - split
    x_train, x_test, y_train, y_test = train_test_split(
        iris_data.data, 
        iris_data.target, 
        test_size = 0.2, 
        random_state = 42
    )
    # Targets are discrete -> Categorization
    # Using StandardScaler
    transfer = StandardScaler()

    # Standardization
    # Feature Engineering
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)

    # Create estimator() as model object 
    estimator = KNeighborsClassifier(n_neighbors = 3)

    # Model Training
    estimator.fit(x_train, y_train)

    # Model Prediction
    y_predict = estimator.predict(x_test)

    # Model Evaluation 
    score = accuracy_score(y_test, y_predict)

    print(f'the model accuracy score is {score}')
```
---
Below We will use Cross Validation and Grid Search to determine K

```python
from sklearn.model_selection import train_test_split, GridSearchCV

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
```
> Again, Emphasis: Once GridSearchCV finishes cross-validation, it automatically stores the best parameters, the best model, and the best score — you don’t need to manually input k or retrain the model yourself.

```python
"""
Results: 
0.95
----
KNeighborsClassifier(n_neighbors=3)
----
{'n_neighbors': 3}
----
{'mean_fit_time': array([0.00032091, 0.00014877, 0.00014812, 0.00013107, 0.00012827,
       0.00013381, 0.00013202, 0.00013089, 0.00013024, 0.00017577]), 'std_fit_time': array([3.07154989e-04, 3.07259487e-05, 1.63246012e-05, 7.60981162e-07,
       9.38654885e-07, 1.12251546e-05, 2.27045977e-06, 2.92488197e-06,
       3.29231369e-06, 4.74675689e-05]), 'mean_score_time': array([0.00042623, 0.00037992, 0.00036538, 0.0003345 , 0.00032824,
       0.00042826, 0.00034249, 0.00034124, 0.00034851, 0.00046474]), 'std_score_time': array([1.11134439e-04, 8.16371291e-05, 2.54128372e-05, 5.90777955e-06,
       4.69138159e-06, 1.55700875e-04, 6.72767213e-06, 5.33752855e-06,
       1.57007052e-05, 9.84468646e-05]), 'param_n_neighbors': masked_array(data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             mask=[False, False, False, False, False, False, False, False,
                   False, False],
       fill_value=999999), 'params': [{'n_neighbors': 1}, {'n_neighbors': 2}, {'n_neighbors': 3}, {'n_neighbors': 4}, {'n_neighbors': 5}, {'n_neighbors': 6}, {'n_neighbors': 7}, {'n_neighbors': 8}, {'n_neighbors': 9}, {'n_neighbors': 10}], 'split0_test_score': array([0.96666667, 0.93333333, 0.96666667, 0.9       , 0.93333333,
       0.93333333, 0.96666667, 0.96666667, 0.96666667, 0.96666667]), 'split1_test_score': array([0.9       , 0.93333333, 0.93333333, 0.9       , 0.9       ,
       0.93333333, 0.93333333, 0.93333333, 0.96666667, 0.96666667]), 'split2_test_score': array([0.93333333, 0.93333333, 0.93333333, 0.93333333, 0.93333333,
       0.93333333, 0.93333333, 0.93333333, 0.93333333, 0.93333333]), 'split3_test_score': array([0.96666667, 0.93333333, 0.96666667, 0.93333333, 0.9       ,
       0.93333333, 0.93333333, 0.93333333, 0.93333333, 0.93333333]), 'mean_test_score': array([0.94166667, 0.93333333, 0.95      , 0.91666667, 0.91666667,
       0.93333333, 0.94166667, 0.94166667, 0.95      , 0.95      ]), 'std_test_score': array([0.02763854, 0.        , 0.01666667, 0.01666667, 0.01666667,
       0.        , 0.01443376, 0.01443376, 0.01666667, 0.01666667]), 'rank_test_score': array([ 4,  7,  1,  9, 10,  7,  4,  4,  1,  1], dtype=int32)}

The accuracy score is 1.0
"""
```