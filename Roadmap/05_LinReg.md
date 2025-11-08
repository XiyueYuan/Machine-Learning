## Linear Regression 
`from sklearn import LinearRegressor`

1. Introduction of Linear Regression
- supervised learning 
- Simple Linear Regression 
$y = wx + b$
    - w: weight
    - b: bias
    - can be used when number of feature == 1

- Multiple Linear Regression
$ y = w_{1}x_{1} + w_{2}x_{2} + w_{3}x_{3} + ...$
in which 
$$
\begin{bmatrix}
1 & x_{11} & x_{12} \\
1 & x_{21} & x_{22} \\
1 & x_{31} & x_{32}
\end{bmatrix}
\begin{bmatrix}
b \\ w_1 \\ w_2
\end{bmatrix}
=
\begin{bmatrix}
\hat{y}_1 \\ \hat{y}_2 \\ \hat{y}_3
\end{bmatrix}
$$

$$
w = \begin{bmatrix}
b \\ w_1 \\ w_2
\end{bmatrix}
x = \begin{bmatrix}
1 \\ x_1 \\ x_2
\end{bmatrix}
$$

2. Model Eval
- MAE: Mean Absolute Error
- MSE: Mean Square Error
- RMSE: Root Mean Square Error

3. Loss Function (Cost Function)
- measures how well a machine learning model predicts the target output
- quantifies the difference between the predicted value $\hat{y}$ and the true value $y$
- minimize the loss function

$L(k, b)$
1. Error = Predicted Value - Real Value
2. Loss Function: A function that measures the difference between the model’s prediction and the true value for each sample.
> The smaller the loss, the better the model fits the data.
- Ordinary Least Square 
In a simple linear regression: 
given a list of `x_train` and `y_train`, to get the optimal w for linear regression `y = wx + b`, we need to assume the intercept b
```python
height = [[160], [166], [172], [174], [180]]
weight = [56.3, 60.6, 65.1, 68.5, 75]
x_test = [[176]]
```
To get the best w, the sum of all squared errors should be minimum. 
- Let's assume b, the intercept is -100
$y = wx - 100$
For each value, the predicted value `height * w - 100`, the real value is `weight`

- Then, 
$MSE$
$ = 1/n * ((160w - 100 - 56.3) ^2 + (166w - 100 - 60.6) ^ 2 + ...)$
$ = 1/n * ( 256000w ^ 2 - 281671.6w + 136496.32)$

 - In order to get the minimum error, we differentiate it.
==> $2 * w * 256000 - 281671.6 = 0$
We get w = 0.55