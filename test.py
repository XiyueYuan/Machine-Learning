from sklearn.datasets import load_iris
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load data
iris = load_iris()
iris_data = load_iris(as_frame = True)
df = pd.DataFrame(iris_data.frame)

# Explortary Data Analysis EDA
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

# Data Visualization
def visualization():
    # generating a pairplot to display the relationship between two features pairwisely
    sns.pairplot(df, hue="target", diag_kind="kde", corner=True)
    plt.show()

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

if __name__ == '__main__': 
    model()