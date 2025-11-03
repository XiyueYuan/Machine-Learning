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
