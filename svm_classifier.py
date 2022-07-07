import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

class MYSVM:
    def __init__(self,lr=.005,iters=1000,lambda_p=0.01):
        self.lr=lr
        self.iters=iters
        self.lambda_p = lambda_p

        self.w = None
        self.b = None
        
    def fit(self,X,y):
        n_samples,n_feat = X.shape
        
        self.w = np.zeros(n_feat)
        self.b = 0
        
        y_true = np.where(y <= 0, -1, 1) # same as -1 if y <= 0 else 1
        
        for _ in range(self.iters):
            for idx, sample in enumerate(X):
                # do a forward pass
                y_pred = y_true[idx]*(np.dot(sample,self.w)-self.b)
                
                # update with the gradient
                if y_pred>=1:
                    self.w -= self.lr * (2 * self.lambda_p * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_p * self.w - np.dot(sample, y_true[idx]))
                    self.b -= self.lr * y_true[idx]
        
    def predict(self,X):
        # do a forward pass
        y = np.dot(X, self.w) - self.b
        return np.sign(y) # if -1, belongs to class 0, else class 1

data = datasets.load_breast_cancer()
X = data.data
y = data.target
y[y==0]=-1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

svm_clf = MYSVM()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy(y_test, y_pred)
acc
