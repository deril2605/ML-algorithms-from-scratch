import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy
  
X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=123)
## Splitting data randomly as train (80% of the data) and test (20% of the data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

class myNB():
    def fit(self,X_train,y_train):
        self.samples,self.features = X.shape
        self._classes = np.unique(y_train)
        classes = len(self._classes)
        
        self._mean = np.zeros((classes,self.features),dtype=np.float64)
        self._var = np.zeros((classes,self.features),dtype=np.float64)
        self._priors = np.zeros(classes,dtype=np.float64)
        
        for idx,c in enumerate(self._classes):
            X_c = X_train[y_train==c]
            self._mean[idx,:] = X_c.mean(axis=0)
            self._var[idx,:] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0]/float(self.samples)
            
    def predict(self,X):
        pred = [self._predict(x) for x in X]
        return np.array(pred)
    
    def _predict(self,x):
        posteriors = []
        
        for idx,c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx,x)))
            posterior += prior
            posteriors.append(posterior)
        return self._classes[np.argmax(posteriors)]
            
    def _pdf(self,idx,x):
        mean = self._mean[idx]
        var = self._var[idx]
        num = np.exp(-(x-mean)**2/(2**var))
        den = np.sqrt(2*np.pi*var)
        return num/den

# instantiate classifier    
mynb = myNB()
## Fitting the model on train data
mynb.fit(X_train,y_train)
## Predict the target's for 20% test data
preds = mynb.predict(X_test)
accuracy(preds,y_test)
