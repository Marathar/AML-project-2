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
    'cap-shape': {'x': 0, 'b': 1, 's': 2, 'f': 3, 'k': 4, 'c': 5},
    'cap-color': {'n': 0, 'y': 1, 'w': 2, 'g': 3, 'e': 4, 'p': 5, 'b': 6, 'u': 7, 'c': 8, 'r': 9},
    'odor': {'p': 0, 'a': 1, 'l': 2, 'n': 3, 'f': 4, 'c': 5, 'y': 6, 's': 7, 'm': 8},
    'gill-attachment': {'f': 0, 'a': 1},
    'gill-spacing': {'c': 0, 'w': 1},
    'stalk-root': {'e': 0, 'c': 1, 'b': 2, 'r': 3, '?': 4},
    'stalk-color-above-ring': {'w': 0, 'g': 1, 'p': 2, 'n': 3, 'b': 4, 'e': 5, 'o': 6, 'c': 7, 'y': 8},
    'stalk-color-below-ring': {'w': 0, 'p': 1, 'g': 2, 'b': 3, 'n': 4, 'e': 5, 'y': 6, 'o': 7, 'c': 8},
    'population': {'s': 0, 'n': 1, 'a': 2, 'v': 3, 'y': 4, 'c': 5}
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