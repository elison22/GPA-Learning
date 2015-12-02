#"""
from sknn import mlp
from sknn.backend import lasagne
from sklearn import neighbors
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from numpy import ndarray
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def exp_trans(x, base):
	return base**x

def norm(x):
	return ((x - np.min(x)) / (np.max(x) - np.min(x)))

allData = pd.read_csv('gpa_real.csv', header = None,
                  	names = ['mobile', 'facebook', 'youtube', 'pinterest', 'netflix',
                  			'tv-movie', 'gaming', 'websurf', 'credits', 'gpa'])

featNames = ['mobile', 'facebook', 'youtube', 'pinterest', 'netflix',
				'tv-movie', 'gaming', 'websurf', 'credits']
labelName = ['gpa']

features = allData[featNames]
labels = allData[labelName]
labels = labels.apply(exp_trans, base=5)
labels = labels.apply(norm)


def runRegressionModel(model):

	# generate indecies for the folds
	folds = KFold(len(allData), n_folds=5)
	curFold = 1
	# this loops over training and prediction for each fold
	for train_rows, test_rows in folds:

		# these four lines split the data into training and testing sets
		trainFeats = features.iloc[train_rows].as_matrix()		
		trainLabs = labels.iloc[train_rows].as_matrix()
		testFeats = features.iloc[test_rows].as_matrix()
		testLabs = labels.iloc[test_rows].as_matrix()

		# train the learner
		model = model.fit(trainFeats, trainLabs)
		
		# measure accuracy
		predictions = model.predict(testFeats)
		predictions = ndarray.flatten(predictions)
		targets = ndarray.flatten(testLabs)
		diffs = targets - predictions
		sse = 0
		se = 0
		denom = len(diffs)
		for val in diffs:
			if str(val) == "nan":
				denom -= 1
				continue
			sse += (val * val)
			se += abs(val)
		print "== Fold " + str(curFold) + " =="
		print "-- ME: " + str( se / denom )
		print "-- MSE: " + str( sse / denom )
		print ""
		curFold += 1

def runKnn():

	# generate indecies for the folds
	folds = KFold(len(allData), n_folds=5)

	# this loops over training and prediction for each fold
	for train_rows, test_rows in folds:	

		# these four lines split the data into training and testing sets
		trainFeats = features.iloc[train_rows]		
		trainLabs = labels.iloc[train_rows]
		testFeats = features.iloc[test_rows]
		testLabs = labels.iloc[test_rows]

		# train the learner
		knn = neighbors.KNeighborsRegressor(5, "distance")
		knn.fit(trainFeats, trainLabs)
		
		# measure accuracy
		predictions = knn.predict(testFeats)
		predictions = ndarray.flatten(predictions)
		targets = ndarray.flatten(testLabs.values)
		diffs = targets - predictions
		sse = 0
		se = 0
		denom = len(diffs)
		for val in diffs:
			if str(val) == "nan":
				denom -= 1
				continue
			sse += (val * val)
			se += abs(val)
		print ( se / denom )
		print ( sse / denom )
		print ""

def testDataTrans():

	someData = allData
	someData = someData[labelName]
	scaledData = someData.apply(exp_trans, base=5)

	norm1 = someData.apply(norm)
	norm2 = scaledData.apply(norm)
	print norm1.values
	print norm2.values

	bothGPAs = pd.concat([norm1, norm2], axis=1)

	# plt.figure()
	norm1.plot(kind='hist', alpha=.5)
	norm2.plot(kind='hist', alpha=.5)
	plt.show()

knn = neighbors.KNeighborsRegressor(5, "distance")
percep = linear_model.Perceptron(n_iter=15)

layers = []
layers.append(mlp.Layer("Sigmoid", units=9))
layers.append(mlp.Layer("Sigmoid", units=18))
layers.append(mlp.Layer("Linear", units=1))
MLP = mlp.Regressor(layers, learning_rule="momentum")

runRegressionModel(knn)
# runRegressionModel()
runRegressionModel(MLP)

"""
	features = allData[featNames]
	labels = allData[labelName]

	# trainFeat, testFeat, trainLabel, testLabel = train_test_split(features, labels, test_size=0.3, random_state=42)


	for train_rows, test_rows in folds:
		roundData = allData.iloc[train_rows]
		print roundData
		# roundTrainLab = labels[train_rows]
		# print "Train: " + str(train_rows) + "  Test: " + str(test_rows)

	knn = neighbors.KNeighborsRegressor(5, "distance")
	knn.fit(trainFeat, trainLabel)
	predictions = knn.predict(testFeat)

	# for idx, pred in enumerate(predictions):
	# 	if(str(pred) == "nan"):
	# 		pred = testLabel[idx]
	# 	print str(pred) + " == " + str(testLabel[idx])
"""


"""
	from sklearn import neighbors
	from mldata import DSReader

	dsr = DSReader()
	ds = dsr.importData('gpa_real.csv')
	ds.splitTrainTest(.7)

	knn = neighbors.KNeighborsRegressor(5, "distance")
	knn.fit(ds.getSet("train-features"), ds.getSet("train-labels"))
	preds = knn.predict(ds.getSet("test-features"))

	for idx, pred in enumerate(preds):
		if(str(pred) == "nan"):
			pred = ds.getSet("test-labels")[idx]
		print str(pred) + " == " + str(ds.getSet("test-labels")[idx])
"""