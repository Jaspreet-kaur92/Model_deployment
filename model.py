import numpy as np
import matplotlib as plt
import pandas as pd
import pickle

dataset = pd.read_csv("hiring.csv")
dataset['Experience'].fillna(0, inplace=True)
dataset["Test_score"].fillna(dataset['Test_score'].mean(),inplace=True)
X= dataset.iloc[:, 1:-1]
print(X)
# Converting words into integer value
def convert_to_int(word):
    word_dict= {'one':1,'two':2, 'three':3,'four':4,"five":5,"six":6,'seven':7,'eight':8,
                'nine':9,'ten':10,'eleven':11,'twelve':12,'zero':0,0:0}
    return word_dict[word]

X['Experience'] = X['Experience'].apply(lambda x : convert_to_int(x))

y = dataset.iloc[:, -1] 

# Splitting training and test set
from sklearn.linear_model import LinearRegression
reg = LinearRegression()

# Fitting the model with training data
reg.fit(X,y)

pickle.dump(reg, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))

print(model.predict([[2,9,6]]))