import numpy as np
import pandas as pd

class LR():
    def __init__(self,lr,itr):
        self.lr=lr
        self.itr=itr
    def fit(self,X,y):
        self.m,self.n=X.shape
        self.w=np.zeros(self.n)
        self.b=0
        self.X=X
        self.y=y
        
        for _ in range(self.itr):
            self.update_weights()
        return self
    def update_weights(self):
        y_pred= self.predict(self.X)
        dW = -2/self.m*((self.X.T).dot(self.y-y_pred))
        db = -2/self.m*(np.sum(self.y-y_pred))
        
        self.w = self.w - self.lr*dW
        self.b = self.b - self.lr*db
        
        return self
    def predict(self,X):
        return np.dot(X,self.w)+self.b
      
mylr = LR(0.05,1000)
mylr.fit(X,y)

from sklearn.metrics import mean_squared_error, mean_absolute_error
mean_squared_error(mylr.predict(X),y,squared=False)
