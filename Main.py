import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import accuracy_score
from sklearn.linear_model import TheilSenRegressor
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import HuberRegressor

​
​
def main():
    Classify()
    # ShowWorthiness()
​
​
def Classify():
    data_train = pd.read_csv('Data/train_copy.csv')
    train = normalize(preprocess(data_train))
​
    Xdata = train.drop(columns='Survived')
​
    ydata = train['Survived']
​
    X_train, X_test, y_train, y_true = train_test_split(Xdata, ydata, test_size=0.1, random_state=42, stratify=ydata)
​
    Class1 = RANSACRegressor(random_state=42)
    Class1.fit(X_train, y_train)
    Class1_predictions = Class1.predict(X_test)
    Class1_accuracy = accuracy_score(y_true, Class1_predictions, normalize=True, sample_weight=None)

    Class2 = TheilSenRegressor(random_state=42)
    Class2.fit(X_train, y_train)
    Class2_predictions = Class1.predict(X_test)
    Class2_accuracy = accuracy_score(y_true, Class2_predictions, normalize=True, sample_weight=None)

    Class3 = LinearRegression()
    Class3.fit(X_train, y_train)
    Class3_predictions = Class3.predict(X_test)
    Class3_accuracy = accuracy_score(y_true, Class3_predictions, normalize=True, sample_weight=None)

    Class4 = HuberRegressor(alpha=0.0, epsilon=epsilon)
    Class4.fit(X_train, y_train)
    Class4_predictions = Class4.predict(X_test)
    Class4_accuracy = accuracy_score(y_true, Class4_predictions, normalize=True, sample_weight=None)

​
​
    print("First Accuracy: ", Class1_accuracy)
    print("Second Accuracy: ", Class2_accuracy)
    print("Third Accuracy: ", Class3_accuracy)
    print("Fourth Accuracy: ", Class4_accuracy)

​
    return
​
​
def ShowWorthiness():
    from sklearn.ensemble import RandomForestClassifier
    data_train = pd.read_csv('Data/train_copy.csv')  # Success!!
    train = normalize(preprocess(data_train))
    Xdata = train.drop(columns='Survived')
    ydata = train[['Survived']]
​
    feat_lables = train.columns[1:]
    forest = RandomForestClassifier(n_estimators=500, random_state=1)
    forest.fit(Xdata, ydata)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(Xdata.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, feat_lables[indices[f]], importances[indices[f]]))
    plt.title('Feature Importance')
    plt.bar(range(Xdata.shape[1]), importances[indices], align='center')
    plt.xticks(range(Xdata.shape[1]), feat_lables, rotation=90)
    plt.xlim([-1, Xdata.shape[1]])
    plt.tight_layout()
    plt.show()
    return
​
​
def simple_plot(data, class0=0, class1=1):
    df = pd.DataFrame({'val1': data[:, 0], 'val2': data[:, 1], 'class': data[:, 2]})
    class1 = df.loc[df['class'] == 1].values
    class0 = df.loc[df['class'] == 0].values
    plt.scatter(class0[:, 0], class0[:, 1], color='red', marker='o', label='class_0')
    plt.scatter(class1[:, 0], class1[:, 1], color='blue', marker='x', label='class_1')
    plt.xlabel('val1')
    plt.ylabel('val2')
    plt.legend(loc='upper left')
    plt.show()
​
​
def preprocess(dataframe):
    dataframe = dataframe.drop(columns=['Cabin', 'Name', 'Ticket'])
    sex_mapping = {label: idx for idx, label in enumerate(np.unique(dataframe['Sex']))}
    dataframe['Sex'] = dataframe['Sex'].map(sex_mapping)
    dataframe = dataframe.rename(columns={'Sex': 'Male'})
    dataframe = pd.get_dummies(dataframe, drop_first=True)
    dataframe = dataframe.rename(columns={'Embarked_Q': 'Q', 'Embarked_S': 'S'})
    dataframe['Survived'] = dataframe['Survived'].replace(0, -1)
    dataframe = dataframe.fillna(dataframe.mean())
    return dataframe
​
​
def normalize(dataframe):
    dataframe = dataframe.set_index('PassengerId')
    normalized_dataframe = (dataframe - dataframe.min()) / (dataframe.max() - dataframe.min())
    return normalized_dataframe
​
​
if __name__ == '__main__':
    main()
