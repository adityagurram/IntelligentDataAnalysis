from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
np.set_printoptions(precision=5, suppress=True)

def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

# Importing the dataset
data = pd.read_csv('HW3-StudentData2.csv')
print(data.shape)
data.head()

# Getting the values and plotting it
d0=data['StudentId'].values;
d1 = data['Phys'].values
d2 = data['Maths'].values
d3=data['English'].values;
d4=data['Music'].values;
X = np.array(list(zip(d1, d2,d3,d4)))
data.iloc[:,1:].plot()
plt.figure();

# Number of clusters
k = 3
# X coordinates of random centroids
C_x = np.random.randint(0, np.max(X)-20, size=k)
# Y coordinates of random centroids
C_y = np.random.randint(0, np.max(X)-20, size=k)
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
print(C)
plt.scatter(C_x, C_y, marker='*', s=200, c='g')

print('cass');
# To store the value of centroids when it updates
C_old = np.zeros(C.shape)
# Cluster Lables(0, 1, 2)
clusters = np.zeros(len(X))
# Error func. - Distance between new centroids and old centroids
error = dist(C, C_old, None)
itr=0;
# Loop will run till the error becomes zero
print('masss');
while error != 0:
    # Assigning each value to its closest cluster
    print(itr);
    for i in range(len(X)):
        distances = dist(X[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster
        print(clusters[i]);
    # Storing the old centroid values
    C_old = deepcopy(C)
    # Finding the new centroids by taking the average value
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    error = dist(C, C_old, None)
    itr=itr+1;
plt.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='r')    