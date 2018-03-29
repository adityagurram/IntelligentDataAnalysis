import numpy as np

fileName="letter-recognition.data.txt"

def getFile(fileName):
		lines = []
		linesWithCommas =open(fileName).read().split('\n')
		for line in (linesWithCommas):
			lines.append(line.split(','))
		return lines

def makeMatrix(lines):
	dataMatrix = np.zeros((len(lines),len(lines[0])-1), dtype=np.float32)
	nameArray = []
	for row, vals in enumerate(lines):
		nameArray.append(lines[row][0])
		dataMatrix[row] = [ int(val) for val in vals[1:] ]	#First row till end of columns in that part. row
	return dataMatrix, nameArray
	
def normalizeValues(datamatrix):
	newMatrix = np.zeros((len(datamatrix),len(datamatrix[0])), dtype=np.float32)
	for column in range(len(datamatrix[0])):
		minVal = np.amin(datamatrix[:,column])
		maxVal = np.amax(datamatrix[:,column])
		denominator = maxVal - minVal
		for row in range(len(datamatrix)):
			newMatrix[row,column] = (datamatrix[row,column] - minVal)/denominator

	return newMatrix

def randomizeData(normalizedMatrix,nameArray):

	traningCount=12000
	validationCount=4000
	testCount=4000
	trainingNameArray=[]
	testNameArray=[]
	validatioNameArray=[]
	
	randArray = np.arange(len(normalizedMatrix))
	np.random.shuffle(randArray)
	traningMatrix = np.zeros((traningCount,len(normalizedMatrix[0])), dtype=np.float32)	
	validationMatrix = np.zeros((validationCount,len(normalizedMatrix[0])), dtype=np.float32)
	testMatrix = np.zeros((testCount,len(normalizedMatrix[0])), dtype=np.float32)
	for iRow in range(len(normalizedMatrix)):
			if(iRow <traningCount):
					traningMatrix[iRow]=normalizedMatrix[randArray[iRow]]
					trainingNameArray.append(nameArray[randArray[iRow]])
			elif (iRow <traningCount +validationCount):
					validationMatrix[iRow-traningCount]=normalizedMatrix[randArray[iRow]]
					validatioNameArray.append(nameArray[randArray[iRow]])
			else:
					testMatrix[iRow-traningCount-validationCount]=normalizedMatrix[randArray[iRow]]
					testNameArray.append(nameArray[randArray[iRow]])
	return trainingNameArray,validatioNameArray,testNameArray,traningMatrix,validationMatrix,testMatrix
				
		
	
	
dataMatrix, nameArray = makeMatrix(getFile(fileName))
normalizedMatrix = normalizeValues(dataMatrix)
trainingNameArray,validatioNameArray,testNameArray,traningMatrix,validationMatrix,testMatrix = randomizeData(normalizedMatrix,nameArray)
print(validatioNameArray,testNameArray,traningMatrix,validationMatrix,testMatrix)

