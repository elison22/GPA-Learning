#print(__doc__)

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Fabian Pedregosa <fabian.pedregosa@inria.fr>
#
# License: BSD 3 clause (C) INRIA


###############################################################################
# Generate sample data

"""
147,10,,,,,,,,,,
"mobile-time", "facebook-time", "youtube-time", "pinterest-time", "netflix-hulu-time", "tv-movie-time", "gaming-time", "websurf-time", "credits", "gpa"
"/home/brandt/Dropbox/ClassesCurrent/CS478/Projects/GroupProject/scikit-testing/gpa_real.csv"
"""

import csv
import random
from sklearn import neighbors

def mainTest():
	features = []
	labels = []
	# features.append([149, 10,'','','','','','','','','',''])
	# features.append(["mobile-time", "facebook-time", "youtube-time", "pinterest-time", "netflix-hulu-time", "tv-movie-time", "gaming-time", "websurf-time", "credits", "gpa"])

	with open('gpa_real.csv', 'rb') as csvfile:
		datareader = csv.reader(csvfile, delimiter=' ', quotechar='|')
		for row in datareader:
			# print ', '.join(row)
			# data.append(row[0].split(','))
			features.append([])
			rowVals = row[0].split(',')
			lastIdx = len(rowVals) - 1
			for idx, val in enumerate(rowVals):
				if idx == lastIdx:
					labels.append(float(val))
				else:
					features[-1].append(float(val))
			# print row[0]

	# print features
	# print len(features)
	# print len(labels)

	sData = splitData(features, labels)

	# print len(sData[0])
	# print len(sData[1])
	# print len(sData[2])
	# print len(sData[3])

	knn = neighbors.KNeighborsRegressor(5, "distance")
	knn.fit(sData[2], sData[3])
	preds = knn.predict(sData[0])

	for idx, pred in enumerate(preds):
		if(str(pred) == "nan"):
			pred = sData[1][idx]
		print str(pred) + " == " + str(sData[1][idx])

def splitData(inFeatures, inLabels):

	toReturn = []
	toReturn.append([])
	toReturn.append([])

	stopSize = int(len(inFeatures)*.7)
	while len(inFeatures) > stopSize:
	
		randIdx = random.random()*len(inFeatures)
		randIdx = int(randIdx)

		toReturn[0].append(inFeatures.pop(randIdx))
		toReturn[1].append(inLabels.pop(randIdx))

	toReturn.append(inFeatures)
	toReturn.append(inLabels)

	return toReturn


mainTest()
