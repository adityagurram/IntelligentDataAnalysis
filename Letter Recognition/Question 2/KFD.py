import random as rd
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
import numpy as np
import graphviz 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from operator import itemgetter

#This function normalise the given input data
def normalizeValues(datamatrix):
	newMatrix = np.zeros((len(datamatrix),len(datamatrix[0])), dtype=np.float32)
	for column in range(len(datamatrix[0])):
		minVal = np.amin(datamatrix[:,column])
		maxVal = np.amax(datamatrix[:,column])
		denominator = maxVal - minVal
		for row in range(len(datamatrix)):
			newMatrix[row,column] = (datamatrix[row,column] - minVal)/denominator
	return newMatrix
	
#This function store data and classifier column separately. 
def BuildDecisionTree():
	Alphabet_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data',sep= ',', header= None)	
	print ("Dataset Dimension  ", Alphabet_data.shape)
	print ("Dataset::")
	Alphabet_data.head()
	X = Alphabet_data.values[:, 0:7]
	Y = Alphabet_data.values[:,8]
	Xnorm=normalizeValues(X)	
	print("the normlaised data is ", Xnorm)		
	return X,Y

#This function uses Grid SearchCV to find top parameter settings
def gridSearchEval(X, y, clftn, parameters_grid, crossval=10):    
    gridSearchData = GridSearchCV(clftn,
                               param_grid=parameters_grid,
                               cv=crossval)
    gridSearchData.fit(X, y)
    print(("\nGrid Search Cross Validation Operation "
           "parameter settings.").format(len(gridSearchData.grid_scores_)))
    parameters_top = present(gridSearchData.grid_scores_, 3)
    return parameters_top	
	
#This function returns the top parameter settings values	
def present(grid_scores, top=3):   
    scores_top = sorted(grid_scores,
                        key=itemgetter(1),
                        reverse=True)[:top]    
    return scores_top[0].parameters

#This function displays best parameter settings 	
def searchgridCV(X_train,y_train):
	print("Grid Parameter Search using 10-fold Cross Validation")
# set of parameters to test
	parameters_grid = {"criterion": ["gini", "entropy"],
              "min_samples_split": [5, 15, 30, 50],
              "max_depth": [None, 5, 15, 30, 50],
              "min_samples_leaf": [5, 15, 30, 50],
              "max_leaf_nodes": [5, 10, 20, 50],
              }			  
	dstree = DecisionTreeClassifier(random_state = 99)
	treesearch_gridsettings = gridSearchEval(X_train, y_train, dstree, parameters_grid, crossval=10)
	print("\n-- Best Parameters:")
	for k, v in treesearch_gridsettings.items():
		print("parameter: {:<20s} setting: {}".format(k, v))
	# test the retuned best parameters
	print("\n\n-- Testing best parameters [Grid]...")
	decisiontree_ts_gs = DecisionTreeClassifier(**treesearch_gridsettings)
	scores_vals = cross_val_score(decisiontree_ts_gs, X_train, y_train, cv=10)
	print("mean: {:.3f} (std: {:.3f})".format(scores_vals.mean(),
                                          scores_vals.std()),
                                          end="\n\n" )
#This function builds the model using 	DecisionTreeClassifier for the given data training data set and predicts the outcome using "Predict function"
#The decision tree graph is store in "dot" format
def EntropyClassification(X_train,y_train,sizeOfData):	
	clf_entropy=[]	
	k_scores=[]		
	clf_entropy=(DecisionTreeClassifier(criterion = "entropy", random_state = 99,min_samples_leaf=sizeOfData))		
	clf_entropy.fit(X_train,y_train)
	tree.export_graphviz(clf_entropy,out_file="KDFDsTree.dot")
	scores=cross_val_score(clf_entropy,X_train,y_train,cv=10,scoring="accuracy")
	print ("The scores for 10 folds are ", scores)
	k_scores.append(scores.mean())	
	return 0
	
#This is the starting point of the program
X_train, y_train=BuildDecisionTree()	
EntropyClassification(X_train,y_train,15)
searchgridCV(X_train,y_train)
clf_entropy_final=[]
clf_entropy_final=(DecisionTreeClassifier(criterion = "entropy",max_depth=5,max_leaf_nodes=50,min_samples_split=5, random_state = 99,min_samples_leaf=15))	
clf_entropy_final.fit(X_train,y_train)
tree.export_graphviz(clf_entropy_final,out_file="KDFDsTree_final.dot")
#data split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.2, random_state = 100)
depth=[] 
for i in range(3,20):
	clf_test = (DecisionTreeClassifier(criterion = "entropy",max_depth=5,min_samples_leaf=15,max_leaf_nodes=50,min_samples_split=5, random_state = 99))	
	clf_test.fit(X_train,y_train)
	scores = cross_val_score(clf_test,X_test, y_test, cv=10,scoring="accuracy")
	depth.append((i,scores.mean()))
#print("The depth is ",depth)
y_pred_en=clf_test.predict(X_test)
CM=confusion_matrix(y_test, y_pred_en)
print ("The confusion Matrix",CM)
a=CM[0][0]
b=CM[0][1]
c=CM[1][0]
d=CM[1][1]
print ("The Accuracy is ", (a+d)/(a+b+c+d))
print ("The Precision is ",(a)/(a+c))
print ("The Recall is ",(a)/(a+b))

