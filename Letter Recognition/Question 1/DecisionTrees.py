#packages required for the program
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

#This function reads the input data file(.csv) and partition the data as Test, Train and Validation using 	train_test_split function
def BuildDecisionTree():
	Alphabet_data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data',sep= ',', header= None)	
	print ("Dataset Dimension  ", Alphabet_data.shape)
	print ("Dataset::")
	Alphabet_data.head()
	X = Alphabet_data.values[:, 1:17]
	Y = Alphabet_data.values[:,0]
	Xnorm=normalizeValues(X)	
	print("the normlaised data is ", Xnorm)
	X_train, X_test, y_train, y_test = train_test_split(Xnorm, Y, test_size = 0.2, random_state = 1)#train: 12000 #test: 4000 #val:4000	
	X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)#validation split		
	return X_train,X_val,X_test, y_train, y_val,y_test
	
#This function builds the model using 	DecisionTreeClassifier for the given data training data set and predicts the outcome using "Predict function"
#The decision tree graph is store in "dot" format
def EntropyClassification(X_train,y_train,sizeOfData,flag,train):	
	clftn_entropy=[]
	clftn_entropy=(DecisionTreeClassifier(criterion = "entropy", random_state = 100,min_samples_leaf=sizeOfData))	
	clftn_entropy.fit(X_train, y_train)	
	if(flag==1):		
		tree.export_graphviz(clftn_entropy,out_file="DsTree.dot")
	if (train==1):
		y_pred_entropy = clftn_entropy.predict(X_val)
	elif (train==2):
		y_pred_entropy = clftn_entropy.predict(X_train)
	elif (train==3):
		y_pred_entropy = clftn_entropy.predict(X_test)
	return y_pred_entropy
	
#This function calculates the accuracy for the predicted outcome values for the given training set.	
def PreditAndPlot(y_pred_entropy,sizeOfData,flag):
	accuracy=0
	if(flag==1):
		accuracy=accuracy_score(y_val,y_pred_entropy)*100
		#print ("Accuracy for data instances "+str(sizeOfData)+" is ",accuracy)
	elif(flag==2):
		accuracy=accuracy_score(y_train,y_pred_entropy)*100	
		#print ("Accuracy for data instances "+str(sizeOfData)+" is ",accuracy)
	return accuracy

#calculates the sum of all the rows elements in a particular column.	
def SumColumn(matrix, col):
	total_sum = 0
	for row in range(len(matrix)):
		if(row != col):
			total_sum = total_sum+ matrix[row][col]
	return total_sum
	
#This function calculates the sum of all the colums data for a particular row.	
def SumRow(matrix, row):
	total_sum = 0
	for col in range(len(matrix)):
		if(row != col):
			total_sum = total_sum+ matrix[row][col]
	return total_sum

#This function calculates Precision , Recall and F metric for the given confusion matrix.	
def CalcPrecRecFmetric(CM,num):
	tp=0 #a
	fp=0 #c	
	tn=0 #d
	fn=0 #b
	tp=CM[num][num]
	fp=SumColumn(CM, num)
	tn=np.trace(CM)-tp
	fn=SumRow(CM,num)
	print ("Calculating Precision, Recall, F Metric for class  ")	
	Precision=(tp)/(tp+fp)
	Recall=(tp)/(tp+fn)
	Fm=(2*(tp))/(2*tp +fn +fp)
	print("Precision for class " +str(chr(num+64))+" is",Precision)
	print ("Recall for class  "+str(chr(num+64))+" is",Recall)
	print("F metric for class "+str(chr(num+64))+" is ",Fm)
	print ("")
	return 0
	
#This is the start point of the program 	
X_train,X_val,X_test, y_train, y_val,y_test=BuildDecisionTree()
PlotArrayX = [250,225,200,175,150,125,100,75,50,25,10,5]
PlotArrayY_Acc=[]
PlotArrayY_Train_Acc=[]

for plot in PlotArrayX:
	y_pred_entropy=EntropyClassification(X_train,y_train,plot,0,1)
	y_pred_train_en=EntropyClassification(X_train,y_train,plot,0,2)
	PlotArrayY_Acc.append(PreditAndPlot(y_pred_entropy,plot,1))
	PlotArrayY_Train_Acc.append(PreditAndPlot(y_pred_train_en,plot,2))
plt.plot(PlotArrayX, PlotArrayY_Acc, 'ro')
plt.plot(PlotArrayX,PlotArrayY_Train_Acc,'rx')
plt.ylabel('Accuracy')
plt.xlabel('Leaf Nodes Data')
plt.show()

length=len(PlotArrayX)
plot=PlotArrayX[length-1]
y_pred_entropy=EntropyClassification(X_train,y_train,plot,1,3)
PreditAndPlot(y_pred_entropy,plot,1)
CM=confusion_matrix(y_test, y_pred_entropy)
print("The confusion matrix is ",CM)
#getting random numbers
nums = [x for x in range(26)]
rd.shuffle(nums)
for iRow in range(3):
	CalcPrecRecFmetric(CM,nums[iRow])