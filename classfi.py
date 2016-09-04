from numpy import *
import matplotlib
import matplotlib.pyplot as plt

#randMat = mat(random.rand(4,4))

#invRandMat = randMat.I

#myEye = randMat * invRandMat

#error = myEye - eye(4)

# print('Hello World!')
# print(error)
# print(eye(4))

# K-Nearest neighbour classfication(kNN)
import kNN
datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.scatter(datingDataMat[:,1],datingDataMat[:,2],
#          15.0*array(datingLabels),15.0*array(datingLabels))
#plt.show()
#fig1 = plt.figure()
#ax1 = fig1.add_subplot(111)
#ax1.scatter(datingDataMat[:,0],datingDataMat[:,1],
#          15.0*array(datingLabels),15.0*array(datingLabels))
#plt.show()
#group,labels = kNN.creatDataSet()
# print (type(group))
# print (type(labels))

#print(kNN.classify0([0,0],group,labels,3))
#print(kNN.classify0([0.9,0.8],group,labels,3))

normMat,ranges,minVals = kNN.autoNorm(datingDataMat)
print(normMat)
print(ranges)
print(minVals)

