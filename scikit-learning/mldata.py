import csv
import random

class DSReader:

	def __init__(self):
		pass

	def importData(self, filepath):
		features = []
		labels = []

		with open(filepath, 'rb') as csvfile:
			datareader = csv.reader(csvfile, delimiter=' ', quotechar='|')
			for row in datareader:
				
				features.append([])
				rowVals = row[0].split(',')
				lastIdx = len(rowVals) - 1
				for idx, val in enumerate(rowVals):
					if idx == lastIdx:
						labels.append(float(val))
					else:
						features[-1].append(float(val))

		return DataSet(features, labels)



class DataSet:

	def __init__(self, features, labels):
		self.trFeatNm = "train-features"
		self.trLabNm = "train-labels"
		self.tstFeatNm = "test-features"
		self.tstLabNm = "test-labels"
		self.dsets = { self.trFeatNm : features, self.trLabNm : labels }

	# parameter: setName = the name of the data group that you want to get. The names are listed in
	# 						the DataSet constructor
	# return: the data group corrosponding to the parameter you passed in.

	def getSet(self, setName):
		return self.dsets[setName]

	def splitData(self, trainRatio, features, labels):
		toReturn = []
		toReturn.append([])
		toReturn.append([])

		stopSize = int(len(features)*trainRatio)
		while len(features) > stopSize:
		
			randIdx = random.random()*len(features)
			randIdx = int(randIdx)

			toReturn[0].append(features.pop(randIdx))
			toReturn[1].append(labels.pop(randIdx))

		toReturn.append(features)
		toReturn.append(labels)
		return toReturn

	def splitTrainTest(self, trainRatio, seed = -1):
		self.dsets[self.tstFeatNm] = []
		self.dsets[self.tstLabNm] = []
		features = self.dsets["train-features"]
		labels = self.dsets["train-labels"]
		if(seed != -1):
			random.seed(seed)

		splitResult = self.splitData(trainRatio, features, labels)

		self.dsets[self.tstFeatNm] = splitResult[0]
		self.dsets[self.tstLabNm] = splitResult[1]

		self.dsets[self.trFeatNm] = splitResult[2]
		self.dsets[self.trLabNm] = splitResult[3]

	def splitToFolds(self, foldCount, foldRatio, seed = -1):
		folds = []
		if(seed != -1):
			random.seed(seed)

		while(len(folds) < foldCount):
			split = splitData
			folds.append([split])

	





dsr = DSReader()
ds = dsr.importData('gpa_real.csv')
ds.splitTrainTest(.7)

print len(ds.getSet("train-features"))