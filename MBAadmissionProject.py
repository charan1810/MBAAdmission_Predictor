#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#load the data
dataset = pd.read_csv(r"C:\Users\CHARAN\Downloads\MBA.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Datapreprocessing
X = dataset.drop(['application_id','gender','international','race','admission'],axis=1).values
X.columns # Index(['gpa', 'major', 'gmat', 'work_exp', 'work_industry'], dtype='object')
#check and fill the missing values if present
selected_rows = np.array(X[:,[0,2,3]],dtype=float)
selected_rows1 = np.array(X[:,[1,4]])
cv=np.any(np.isin(selected_rows1,['nan','NaN','']))
nv = np.any(np.isnan(selected_rows))
print(nv,cv)
#preprocess  major,work_industry column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler,LabelEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1,4])],remainder='passthrough')
X = np.array(ct.fit_transform(X))
X = X.toarray()
le = LabelEncoder()
y=le.fit_transform(y)
#split the dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)
#feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#svm classifier prepn
from sklearn.svm import SVC
classifier = SVC(kernel='rbf',random_state=0)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
#print comparision table
comparison = np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1)
print(comparison)
#check the model confusion matrix and the accuracy score
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test,y_pred)
acs = accuracy_score(y_test, y_pred)
print('confusion matrix: \n',cm)
print('accuracy score: ',acs)
