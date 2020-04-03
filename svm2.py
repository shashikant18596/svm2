import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
digits = load_digits()
df = pd.DataFrame(digits.data,digits.target)
df['target'] = digits.target
x = df.drop('target',axis = 'columns')
y = df['target']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size = 0.2)
model = SVC()
model.fit(x_train,y_train)
model.predict(x_test)
model.score(x_test,model.predict(x_test))
