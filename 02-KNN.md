### KNN Algorithm
##### K-Nearest Neighbors
1. *'k' meaning*
- k means the number of __nearest neighbors__ the algorithm looks at when making a prediction
- if k is too large, resulting in under-fitting
- if k is too small, resulting in over-fitting

- __Euclidean distance__ 
$$
d=\sqrt{\sum_{k=1}^{n}​{(x_{test} - x_{train})^ 2}}
$$
- __Manhattan distance__
$$
d = \sum_{k=1}^{n}|x_{test} - x_{train}|
$$

2. *algorithm*
- classification 
    - discrete
    - by vote
- regression 
    - continuous
    - average

3. Feature preprocessing 
    1. Normalization: to compress features with different scales into the same range (usually between 0 and 1)

    $x' = \frac{x - x_{min}}{max - min}$

    2. Standardization 

    $x' = \frac{x−μ}{σ}​$

4. Cross Validation and Grid Search 

- Used to determine hyperparameter 
- Cross Validation(cv)
    - k-fold cross validation -> mean model accuracy score 
    - used for model eval and hyperparameter tuning 
- Grid Search 
    - help find optimized hyperparameter (K in KNN)
    - Provide a list of parameters → run cross-validation for each parameter → compare the average validation scores → select the best-performing parameter (e.g., the optimal k) → obtain the final model!
`sklearn.model_selection.GridSearchCV(estimator, param_grid = None, cv = None)`
- estimator: estimator object
- cv: k folds
- param_grid: estimator parameter 

> Essentially, it just means running the model multiple times.