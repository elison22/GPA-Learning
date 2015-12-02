import numpy as np
from sklearn import datasets, linear_model
import pandas as pd
from pandas import DataFrame, Series
from sklearn.cross_validation import train_test_split

data = pd.read_csv('gpa_real.csv', header = None,
                  names = ['mobile', 'facebook', 'youtube', 'pinterest', 'netflix',
                  			'tv-movie', 'gaming', 'websurf', 'credits', 'gpa'])
print(data.shape)
data = data.dropna()
data.head()
indep_vars = ['mobile', 'facebook', 'youtube', 'pinterest', 'netflix',
				'tv-movie', 'gaming', 'websurf', 'credits']
dep_vars = ['gpa']
indep_data = data[indep_vars]
dep_data = data[dep_vars]
indep_train, indep_test, dep_train, dep_test = train_test_split(indep_data, dep_data, test_size=0.3)
print dep_train.size
print dep_test.size

print indep_train



