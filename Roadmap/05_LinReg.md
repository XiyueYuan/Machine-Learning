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