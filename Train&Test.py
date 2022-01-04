import pandas as pd
import matplotlib.pyplot as plt

dset = pd.read_csv('prices_en.csv')

plt.scatter(dset['size'], dset['price'])
plt.scatter(dset['market'], dset['price'])

X = dset[['size', 'market']]
Y = dset['price']

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=10)
X_train

len(X_test)
from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(X_train, Y_train)
clf.predict(X_test)

Y_test


