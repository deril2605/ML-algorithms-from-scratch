import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X,y=iris.data,iris.target
X_train,X_test,y_train,y_test = train_test_split(X,y)

def euclidean_distance(x1,x2):
    return np.sqrt(np.sum(x1-x2)**2)

class myKNN():
    def __init__(self,k=3):
        self.k=k
    
    def fit(self,X,y):
        self.Xt=X
        self.yt=y
        
    def predict(self,X):
        predicted_labels = [self._predict(x) for x in X]
        return predicted_labels
      
    def _predict(self,x):
        distances = [euclidean_distance(x,xt) for xt in self.Xt]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest = [self.yt[i] for i in k_indices]
        most_common = Counter(k_nearest).most_common(1)
        return most_common[0][0]
      
knn=myKNN(k=3)
knn.fit(X_train,y_train)
preds=knn.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(preds,y_test)
