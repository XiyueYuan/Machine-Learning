### MACHINE LEARNING
##### Chapter 1: Introduction 
1. Three Big Concepts:
- AI: Artificial Intelligence
    - simulate human intelligence
- ML: Machine Learning
    - field of study that gives computers the ability to learn without being explicitly programmed
- DL: Deep Learning 
    - field of machine learning
2. Train and Test
`X Train`: training features (inputs) used to train the model
`Y Train`: Training labels (the correct answers)
`X Test`: Testing input features
`Y test`: Testing labels
- Process: 
`X_train` → (learn with `y_train`) → model → `X_test` → predict `y_pred` → compare with `y_test`
- Explanation: 
    1. `x_train`(independent variable) and `y_train`(dependent variable) are paird data for computer to "study"
    2. then `x_test` is given without answers, after generating a new output, we compare it with `y_test` 
    3. `x_test` & `x_train` are both features, `y_test` & `y_label` are targets/labels, each row is called a sample
    4. `x_train` & `y_train`: `x_test` & `y_test` ~ 8:2/7:3, 80% of train, 20% of test
3. Supervised Learning and Unsupervised Learning
- __Supervised Learning__ = label + feature
    - if label :: continuous -> regression 
    - if label :: discrete -> classification 

*EX1*. "Prediction of house price"
$Price = f(Area, Location, Floor, ...)$, because house price(label) is continuous, therefore this is a regression model

*EX2*. "Labeling the image"
$Label = f(image)$, because the label is discrete categorization, therefore this is a classfication model

- __Unsupervised Learning__ = feature 
Group or organize the data based on similarity between samples.

*EX3*. "Group the features"
Features = age, income, height, spending habits..
Unsupervised learning algorithm -> 
- cluster1: young, high-spending customers
- cluster2: old, low-spending customers

__Semi-supervised Learning__
- Uses a small amount of labeled data and a large amount of unlabeled data together during training 
- It sits between supervised and unsupervised learning
- Lower labeling costs

__Reinforcement Learning__
- finding the optimal shortest path to get the most rewards
- will talk more in later chapters

4. Machine Learning model process
Data Processing (Data Collection, cleaning, integration..)
->
Feature Engineering (Feature Selection, extraction, ..)
-> 
Model Training (Linear, Decision Tree, SVM, GBDT, ..)
-> 
Model Eval (Regression, MAE, MSE, ..)

5. Feature Engineering 
- Feature extraction 
- Feature preprocessing
- Feature decomposition
- Feature selection
- Feature crosses
6. Fitting
- __under-fitting__: 
    - The model is too simple to capture the underlying pattern in the data.
    - performs poorly on both training and test sets
- __over-fitting__: 
    - learns too much noise or irrelevant details from the training data
    - performs very well on the training set but poorly on the test set
- Criteria: __generalization ability__
7. Model selection 
<p align="center">
<img src= 'assets/roadmap.png'
width=""/>
</p>