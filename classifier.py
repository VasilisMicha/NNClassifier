from sklearn.metrics import accuracy_score
import scipy.io
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
import time

test = scipy.io.loadmat('../test.mat')
train = scipy.io.loadmat('../train.mat')

Xtrain = train['X']
ytrain = np.ravel(train['y'])
Xtest = test['X']
Ytrue = np.ravel(test['y'])

print(Xtrain.shape)
print(ytrain.shape)

def normalize(x):
    return x/255 # deep learning models prefer number [0, 1]

Xtrain = np.transpose(Xtrain, (3, 0, 1, 2))
Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1] * Xtrain.shape[2] * Xtrain.shape[3]))
Xtrain = np.array([normalize(xi) for xi in Xtrain])

Xtest = np.transpose(Xtest, (3, 0, 1, 2))
Xtest = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[1] * Xtest.shape[2] * Xtest.shape[3]))
Xtest = np.array([normalize(xi) for xi in Xtest])

start = time.time()

#neigh = KNeighborsClassifier(n_neighbors=1).fit(Xtrain, ytrain)
#Ypred = neigh.predict(Xtest)
   
#neigh = KNeighborsClassifier(n_neighbors=3).fit(Xtrain, ytrain)
#Ypred = neigh.predict(Xtest)

clf = NearestCentroid().fit(Xtrain, ytrain)
Ypred = clf.predict(Xtest)

end = time.time()
print("time: "+str(end - start))

print(accuracy_score(Ytrue, Ypred))

