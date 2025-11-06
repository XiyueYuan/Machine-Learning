### MNIST Dataset: 
- a large collection of handwritten digits (0-9) commonly used for training and testing machine learning and computer vision models

### Steps: 
1. Import packages (sklearn)
2. Load Data + EDA (pandas, matplotlib)
3. Data Preprocessing (split)
4. Feature Engineering (Standardization / Normalization)
5. Model Training (estimator = MODEL)
6. Model Prediction (y_predict = estimator.predict())
7. Model Evaluation (accuracy_score)

#### First we take a peek at the data shape
```python
import pandas as pd 
df = pd.read_csv('../assets/data/archive/mnist_train.csv')
# Take a peek at the data shape
print(df.shape) 
print(df.iloc[0:5, 0:10])

"""
(60000, 785)
   label  1x1  1x2  1x3  1x4  1x5  1x6  1x7  1x8  1x9
0      5    0    0    0    0    0    0    0    0    0
1      0    0    0    0    0    0    0    0    0    0
2      4    0    0    0    0    0    0    0    0    0
3      1    0    0    0    0    0    0    0    0    0
4      9    0    0    0    0    0    0    0    0    0
"""
```
- We get the result: the dataset is composed of 60000 rows and 785 columns. Each row (case) represents each `x_train`, the `y_train`: label is listed at the first column.
- Each picture is flatten to a 28 * 28 row, 28 * 28 = 785 - 1

#### Import Packages 
```python
import matplotlib.pyplot as plt
import joblib # used to save the model
import pandas as pd
from sklearn.model_selection import train_set_split
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from sklearn.metrics import accuracy_score
```

#### Load Data + EDA
- Manually split the data from the (60000, 785) dataset
```python
x_train = df.iloc[:, 1:].values
y_train = df.iloc[:, 0].values
print(Counter(y_train))
print(x_train.shape)
print(y_train.shape)

"""
y_train = df.iloc[:, 0].values
x_train = df.iloc[:, 1:].values

# Here we can notice the category of label: 9
print(Counter(y_train))
print(x_train.shape)
print(y_train.shape)
"""
```
#### Model Training
```python
estimator = KNeighborsClassifier(n_neighbors = 3)
estimator.fit(x_train, y_train)
joblib.dump(estimator, 'cd')
# estimator = joblib.load('cd')
```
#### Model Eval
```python
estimator.score(x_test, y_test)
```
