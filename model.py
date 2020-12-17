import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pylab as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from numpy.random import *
from sklearn import linear_model


X_train = rand(100,2)
X_test = rand(100,2)


w1 = 1.0
w2 = 2.0
b =3.0

noise_train = 0.1*randn(100)
noise_test = 0.1*randn(100)

Y_train = w1*X_train[:,0] + w2*X_train[:,1] + b + noise_train
Y_test = w1*X_test[:,0] + w2*X_test[:,1] + b + noise_test



linear_reg_model = linear_model.LinearRegression()

linear_reg_model.fit(X_train, Y_train)


Y_pred = linear_reg_model.predict(X_test)

result = pd.DataFrame(Y_pred)
result.columns = ['Y_pred']
result['Y_test'] = Y_test
sns.set_style('darkgrid')
sns.regplot(x='Y_pred', y='Y_test', data=result)
plt.show()