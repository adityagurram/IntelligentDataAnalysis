import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from sklearn.decomposition import PCA
 #compute rand index for cluster 2 and cluster 3
def computeRandIndex(A,B,rCount):        
        a=0;
        b=0;
        c=0;
        d=0;
        for i in range(rCount):
            for j in range(i+1,rCount):
                if(A[i]==A[j] and B[i]==B[j]):
                    a=a+1;
                elif(A[i]!=A[j] and B[i]==B[j]):
                    b=b+1;
                elif (A[i]!=A[j] and B[i]!=B[j]): 
                    d=d+1;
                else:
                    c=c+1; 
        print('Rand Index A, B , C , D values down here ');                    
        print('a,b,c,d',a,b,c,d);            
        randIndex=(a+d)/(a+b+c+d);
        return randIndex;
def calculateAvg(index_Range,X):
        C=[];
        PhsData=0;
        MathsData=0;    
        EngData=0;
        MusicData=0;
        count=0;
        for idx in index_Range:        
            PhsData=PhsData+X[idx,0];
            MathsData=MathsData+X[idx,1];
            EngData=EngData+X[idx,2];
            MusicData=MusicData+X[idx,3];
            count=count+1;
        avgPhy=PhsData/count;
        avgMat=MathsData/count;
        avgEng=EngData/count;
        avgMusic=MusicData/count;
        C=[avgPhy,avgMat,avgEng,avgMusic]    
        return C;
def Hierarchical():        
    plt.rcParams['figure.figsize'] = (16, 9)
    plt.style.use('ggplot')
    np.set_printoptions(precision=5, suppress=True)
    
    # Importing the dataset
    data = pd.read_csv('HW3-StudentData2.csv')
    print(data.shape)
    data.head()
    
    # Getting the values and plotting it    
    d1 = data['Phys'].values
    d2 = data['Maths'].values
    d3=data['English'].values;
    d4=data['Music'].values;
    X = np.array(list(zip(d1, d2,d3,d4)))  
    data.iloc[:,1:].plot()
    plt.figure();
    
    #clustering method
    Z_single = linkage(X, 'single')
    Z_complete = linkage(X, 'complete')
    c_single, coph_dists_single = cophenet(Z_single, pdist(X));
    c_complete, coph_dists_complete = cophenet(Z_complete, pdist(X));
    
    # calculate full dendrogram
    f1 = plt.figure(figsize=(25, 10))
    ax1 = f1.add_subplot(111)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        Z_single,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8., 
        # font size for the x axis labels
    )
    f2 = plt.figure(figsize=(25, 10))
    ax2 = f2.add_subplot(111)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        Z_complete,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
    )
    plt.show()
    
    max_c = 4
    clusters_single = fcluster(Z_single, max_c, criterion='maxclust')
    print('Single Link Cluster Composition');
    print(clusters_single[0:10])
    print(clusters_single[10:20])
    print(clusters_single[20:30])
    print(clusters_single[30:40])
    print(clusters_single[50:60])
    print(clusters_single[60:69])
    
    
    max_c = 4
    clusters_complete = fcluster(Z_complete, max_c, criterion='maxclust')
    print('Complete Link Cluster Composition');
    print(clusters_complete[0:10])
    print(clusters_complete[10:20])
    print(clusters_complete[20:30])
    print(clusters_complete[30:40])
    print(clusters_complete[50:60])
    print(clusters_complete[60:69])
    
    index_single_one =[i for i, x in enumerate(clusters_single) if x == 1]
    index_single_two =[i for i, x in enumerate(clusters_single) if x == 2]
    index_single_three =[i for i, x in enumerate(clusters_single) if x == 3]
    index_single_four =[i for i, x in enumerate(clusters_single) if x == 4]
    
    centroid_single_one=calculateAvg(index_single_one,X);
    centroid_single_two=calculateAvg(index_single_two,X);
    centroid_single_three=calculateAvg(index_single_three,X);
    centroid_single_four=calculateAvg(index_single_four,X);
    print('single link centroids for 4 clusters');
    print(centroid_single_one);
    print(centroid_single_two);
    print(centroid_single_three);
    print(centroid_single_four);
    
    print('')
    print('')
    #complete link
    
    index_complete_one =[i for i, x in enumerate(clusters_complete) if x == 1]
    index_complete_two =[i for i, x in enumerate(clusters_complete) if x == 2]
    index_complete_three =[i for i, x in enumerate(clusters_complete) if x == 3]
    index_complete_four =[i for i, x in enumerate(clusters_complete) if x == 4]
       
    centroid_complete_one=calculateAvg(index_complete_one,X);
    centroid_complete_two=calculateAvg(index_complete_two,X);
    centroid_complete_three=calculateAvg(index_complete_three,X);
    centroid_complete_four=calculateAvg(index_complete_four,X);
    print('complete link centroids for 4 clusters');
    print(centroid_complete_one);
    print(centroid_complete_two);
    print(centroid_complete_three);
    print(centroid_complete_four);
    
    rows=clusters_single.shape;
    rCount=rows[0];
    result=computeRandIndex(clusters_single,clusters_complete,rCount);
    print('rand index for single and complete link ',result);
    return clusters_single;
    