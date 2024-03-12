import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

current_dir = os.path.dirname(os.path.abspath(__file__))
data = pd.read_csv(os.path.join(current_dir, "./data/mushrooms.csv"))
col_names = data.columns
############# The chosen data #############
# Define mappings for each category
mappings = {
    'class': {'e': 0, 'p': 1},
    'cap-shape': {'b': 0, 'c': 1, 'x': 2, 'f': 3, 'k': 4, 's': 5},
    'cap-surface': {'f': 0, 'g': 1, 'y': 2, 's': 3},
    'cap-color': {'n': 0, 'b': 1, 'c': 2, 'g': 3, 'r': 4, 'p': 5, 'u': 6, 'e': 7, 'w': 8, 'y': 9},
    'bruises': {'t': 0, 'f': 1},
    'odor': {'a': 0, 'l': 1, 'c': 2, 'y': 3, 'f': 4, 'm': 5, 'n': 6, 'p': 7, 's': 8},
    'gill-attachment': {'a': 0, 'd': 1, 'f': 2, 'n': 3},
    'gill-spacing': {'c': 0, 'w': 1,'d' :2},
    'gill-size': {'b': 0, 'n': 1},
    'gill-color': {'k': 0, 'n': 1, 'b': 2, 'h': 3, 'g': 4, 'r': 5, 'o': 6, 'p': 7, 'u': 8, 'e': 9, 'w': 10, 'y': 11},
    'stalk-shape': {'e': 0, 't': 1},
    'stalk-root': {'b': 0, 'c': 1, 'u': 2, 'e': 3, 'z': 4,'r': 5, '?': 6},
    'stalk-surface-above-ring': {'f': 0, 'y': 1, 'k': 2, 's':3},
    'stalk-surface-below-ring': {'f': 0, 'y': 1, 'k': 2, 's':3},
    'stalk-color-above-ring': {'n': 0, 'b': 1, 'c': 2, 'g': 3, 'o': 4, 'p': 5, 'e': 6, 'w': 7, 'y': 8},
    'stalk-color-below-ring': {'n': 0, 'b': 1, 'c': 2, 'g': 3, 'o': 4, 'p': 5, 'e': 6, 'w': 7, 'y': 8},
    'veil-type': {'p': 0, 'u': 1},
    'veil-color' : {'n': 0, 'o': 1, 'w': 2, 'y': 3},
    'ring-number': {'n': 0, 'o': 1, 't': 2},
    'ring-type': {'c': 0, 'e': 1, 'f': 2, 'l': 3, 'n': 4, 'p': 5, 's': 6, 'z': 7},
    'spore-print-color': {'k': 0, 'n': 1, 'b': 2, 'h': 3, 'r': 4, 'o': 5, 'u': 6, 'w': 7, 'y': 8},
    'population': {'a': 0, 'c': 1, 'n': 2, 's': 3, 'v': 4, 'y': 5},
    'habitat': {'g': 0, 'l': 1, 'm': 2, 'p': 3, 'u': 4, 'w': 5, 'd': 6}
}
# Apply mappings to the dataset
for category, mapping in mappings.items():
    data[category] = data[category].map(mapping)

X = data.drop('class', axis=1)
y = data['class']

Xtest = X[:1000]
ytest = y[:1000]
Xpool = X[1000:]
ypool = y[1000:]

model = LogisticRegression()
np.random.seed(42)
addn = 2
order=np.random.permutation(range(len(Xpool)))

poolidx=np.arange(len(Xpool),dtype='int')
ninit = 10

trainset=order[:ninit]

Xtrain=np.take(Xpool,trainset,axis=0)
ytrain=np.take(ypool,trainset,axis=0)

poolidx=np.setdiff1d(poolidx,trainset)

testacc=[]
for i in range(25):
    model.fit(np.take(Xpool,order[:ninit+i*addn],axis=0),np.take(ypool,order[:ninit+i*addn],axis=0))
    #predict and calculate the accuracy
    acc = model.score(Xtest,ytest)
    #calculate accuracy on test set
    testacc.append((ninit+i*addn,acc)) #add in the accuracy
    print('Model: LR, %i random samples'%(ninit+i*addn))

print("idk man")