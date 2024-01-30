import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv(r'C:\Users\Home\Desktop\parkinson.csv.zip')


X = df.drop(columns=['name' , 'status'],axis=1)
Y = df['status']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=2)

ss = StandardScaler()
ss.fit(X_train)

X_train = ss.transform(X_train)
X_test = ss.transform(X_test)

model = svm.SVC(kernel = 'linear')

model.fit(X_train,Y_train)

X_train_pred = model.predict(X_train)
train_data_acc = accuracy_score(Y_train,X_train_pred)
print("Accuracy of training data : ",train_data_acc)

X_test_pred = model.predict(X_test)
test_data_acc = accuracy_score(Y_test,X_test_pred)
print("Accuracy of testing data : ",test_data_acc)

input_data =(110.56800,125.39400,106.82100,0.00462,0.00004,0.00226,0.00280,0.00677,0.02199,0.19700,0.01284,0.01199,0.01636,0.03852,0.00420,25.82000,0.429484,0.816340,-5.391029,0.250572,1.777901,0.232744)
input_data_np = np.asarray(input_data)
input_data_re = input_data_np.reshape(1,-1)
s_data = ss.transform(input_data_re)
pred = model.predict(s_data)
print(pred)
if pred[0]==0:
    print("Negative")
else:
    print("Positive")