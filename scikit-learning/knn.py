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