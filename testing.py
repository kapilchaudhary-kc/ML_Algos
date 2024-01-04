import matplotlib.pyplot as plt
import numpy as np
from KMeans_Clustering import KMeansClustering
random_points=np.random.randint(0,100,size=(100,2))
random_points=np.array(random_points)
kmeans=KMeansClustering(k=3)
labels=kmeans.fit(random_points)
plt.scatter(random_points[:,0],random_points[:,1],c=labels)
plt.scatter(kmeans.centroids[:,0],kmeans.centroids[:,1],c=range(len(kmeans.centroids)),marker="*",s=200)
plt.savefig("Result.png")
plt.show()