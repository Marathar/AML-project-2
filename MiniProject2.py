import os
import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
data = pd.read_csv(os.path.join(current_dir, "./data/mushrooms.csv"))
col_names = data.columns
# print(data.shape)
# print(col_names)

X = data.drop('class', axis=1)
y = data['class']

