# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 01:07:19 2019

@author: Nan
"""

# 利用KNN分类算法进行分类
from sklearn import neighbors, cluster, svm, datasets

knn=neighbors.KNeighborsClassifier()
svc = svm.LinearSVC()
iris=datasets.load_iris()

#从已有数据中学习
knn.fit(iris.data,iris.target)
 #判断萼片长度和宽度、花瓣长度和宽度分别是5.0cm, 3.0cm, 5.0cm, 2.0cm的鸢尾花所属类别。
knn.predict([[5,4,5,2]])
a=knn.predict([[5,4,5,2]])



#利用k-means聚类算法进行聚类
#from sklearn import cluster, datasets
#from skelearn.cluster import KMeans
#iris=datasets.load_iris()
kmeans=cluster.KMeans(n_clusters=3).fit(iris.data)
#确定数据的类型
#pred=kmeans.predict(iris.data)   
pred=kmeans.predict([[5,4,5,2]])  


#利用svm聚类算法进行聚类
#from sklearn import svm, datasets
#iris = datasets.load_iris()

print (a, pred)

#print (iris.values())

'''

svc.fit(iris.data, iris.target) # 学习
svc.predict([[ 5.0, 3.0, 5.0, 2.0]])  # 预测
b=svc.predict([[ 5.0, 3.0, 5.0, 2.0]])  
print(a,pred,b)

'''

