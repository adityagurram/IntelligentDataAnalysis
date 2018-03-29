from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_samples, silhouette_score
from matplotlib import pyplot as plt
from Hierarchical import Hierarchical,computeRandIndex
from sklearn.decomposition import PCA
def computeIndividualSSE (k_cluster,labels,centroids,X):
    Indi_SSE=[];    
    for Kval in range(0,k_cluster):        
        indices = [i for i, l in enumerate(labels) if l == Kval]        
        A0=np.array(np.take(X[:,0],indices));        
        A1=np.array(np.take(X[:,1],indices));       
        A=np.column_stack((A0, A1))       
        C_temp=centroids[Kval,:];
        B0=np.array(np.tile(C_temp[0],len(indices)));       
        B1=np.array(np.tile(C_temp[1],len(indices)));                 
        B=np.column_stack((B0, B1));         
        indSSE=((A-B)**2).sum();        
        Indi_SSE.append(indSSE);                     
    return Indi_SSE;
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
np.set_printoptions(precision=5, suppress=True)

# Importing the data
data = pd.read_csv('HW3-StudentData2.csv')
print('The data dimensions',data.shape)
data.head()

# Getting the values and plotting it
d0=data['StudentId'].values;
d1 = data['Phys'].values
d2 = data['Maths'].values
d3=data['English'].values;
d4=data['Music'].values;
X = np.array(list(zip(d1, d2,d3,d4)))

#PCA analysis for dimensionality reduction
pca=PCA(n_components=2)
pca.fit(X);
print('PCA variance ratio is ',pca.explained_variance_ratio_);
X=pca.transform(X);

data.iloc[:,1:].plot()
plt.figure();
TotalSSE=0;
#final
Fcentroids=[];
FSSE=[];
Fdistortions=[];
Findi_clusters=[];
FSilh_avg=[];
Flabels=[];
Karray=[3,4,5,6,7,8];
itr=-1;
for i in Karray:
    itr=itr+1;
    JSilh_avg=[];
    JSSE=[];
    Jdistortions=[];
    Jindi_clusters=[];
    Jlabels=[];
    Jcentroids=[];
    for j in range(0,3):  #his is for running 3 times for each K value         
        # Number of clusters    
        kmeans = KMeans(n_clusters=i)
        # Fitting the input data
        kmeans = kmeans.fit(X) 
            # Centroid values
        centroids = kmeans.cluster_centers_         
        if(j==0 or JSSE[0] >=kmeans.inertia_):            
            JSilh_avg=[];
            JSSE=[];
            Jdistortions=[];
            Jindi_clusters=[];
            Jlabels=[];
            Jcentroids=[];
        # Getting the cluster labels
            labels=kmeans.predict(X);
            Jlabels.append(kmeans.predict(X));
            Jindi_clusters.append(computeIndividualSSE(i,labels,centroids,X));    
            JSSE.append(kmeans.inertia_);
            #print('JSSE size',len(JSSE));
            print('clustering centers k =',i,'  Iteration is ',j);
            print('Total SSE ',JSSE);
            print('centroids ',centroids);
            print('Individual clusters SSE values are',Jindi_clusters)
            silhouette_avg = silhouette_score(X, labels);
            JSilh_avg.append(silhouette_avg);   
            Jcentroids.append(centroids);
            Jdistortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])            
    Flabels.append(Jlabels); 
    Fcentroids.append(Jcentroids);
    Findi_clusters.append(Jindi_clusters);
    FSSE.append(JSSE);
    Fdistortions.append(Jdistortions);
    #print('FSSE shape',len(FSSE));
    FSilh_avg.append(JSilh_avg);   
    print(' ');
    print(' ');
    print ('Selected Iteration Details for clusters',i);
    print('The Final lables are ',Flabels[itr]);
    print('The final cluster centers are',Fcentroids[itr]);
    print('Individual clusters SSE values are',Findi_clusters[itr]);
    print('Final Total SSE is ',FSSE[itr]);
    print('The Final average silhouette_score is :', FSilh_avg[itr]);
    print(' ');
    print(' ***********************************************');
    
f1 = plt.figure()
f2 = plt.figure()
f3 = plt.figure()
ax1 = f1.add_subplot(111)
ax1.plot(Karray,FSSE)
ax2 = f2.add_subplot(111)
ax2.plot(Karray,FSilh_avg)
ax3 = f3.add_subplot(111)
plt.plot(Karray, Fdistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

print('SO whats the best clusters needed?????')
selectedkindex=FSilh_avg.index(max(FSilh_avg));
selectedkval=Karray[FSilh_avg.index(max(FSilh_avg))];
print('The best clustering is selcted based on Silhoutte coefficeient and the value is ',selectedkval)
print('The Final lables are ',Flabels[selectedkindex]);
print('The final cluster centers are',Fcentroids[selectedkindex]);
print('Individual clusters SSE values are',Findi_clusters[selectedkindex]);
print('Final Total SSE is ',FSSE[selectedkindex]);
print('The Final average silhouette_score is :', FSilh_avg[selectedkindex]);
print(' ');
print(' ***********************************************');
    


#generating random data points
D1=np.random.uniform(low=0,high=100,size=100)
D1=np.int_(D1);

D2=np.random.uniform(low=0,high=100,size=100)
D2=np.int_(D2);

D3=np.random.uniform(low=0,high=100,size=100)
D3=np.int_(D3);

D4=np.random.uniform(low=0,high=100,size=100)
D4=np.int_(D4);


NewX = np.array(list(zip(D1, D2,D3,D4)));
#PCA analysis for dimensionality reduction
pcaNew=PCA(n_components=2)
pcaNew.fit(NewX);
print('PCA variance ratio is ',pcaNew.explained_variance_ratio_);
NewX=pcaNew.transform(NewX);
kmeansNew = KMeans(n_clusters=selectedkval)
kmeansNew = kmeansNew.fit(NewX)
labelsNew = kmeansNew.predict(NewX)
centroidsNew = kmeansNew.cluster_centers_
SSENew=kmeansNew.inertia_;
NewIndivudualSSE=computeIndividualSSE(selectedkval,labelsNew,centroidsNew,NewX)
print('clustering centers k =',selectedkval,' for random data points');
print('RD-The Final lables are ',labelsNew);
print('RD-centroids ',centroidsNew);
print('RD- Indiv SSE values',NewIndivudualSSE);
print ('RD-Total SSE',SSENew);
print('Silhoutte Coefficient',silhouette_score(NewX, labelsNew));
print('*******************************');

clusters_single=Hierarchical();
rows=np.array(Flabels[selectedkval]).shape;
rCount=rows[1];
result=computeRandIndex(clusters_single,np.array(Flabels[selectedkval]).T,rCount);
print('rand index for single link and Kmeans clustering',result);


    
    



