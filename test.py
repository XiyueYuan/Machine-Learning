from sklearn.linear_model import LinearRegression
# Features
x_train = [[160], [166], [172], [174], [180]]

# Labels
y_train = [56.3, 60.6, 65.1, 68.5, 75]

# Features 
x_test = [[176]]

estimator = LinearRegression()
estimator.fit(x_train, y_train)
print(estimator.predict(x_test))