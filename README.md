# Tumar_Case

import pandas as pd
import numpy as np

tumar_data = pd.read_csv(r"Downloads\primary-tumor.data",header = None ,delimiter = " *, *",engine= "python")
tumar_data.head()

tumar_data.shape

tumar_data.columns = ["class","age","sex","histologic-type","degree-of-diffe","bone","bone-marrow","lung","pleura","peritoneum","liver","brain","skin","neck","supraclavicular","axillar","mediastinum","abdominal"]
tumar_data.head()

# checking for the null values
tumar_data.isnull().sum()


tumar_data = tumar_data.replace(["?"],np.nan)
tumar_data.head()

tumar_data.dtypes
tumar_data.isnull().sum()

colnames = ["sex","histologic-type","degree-of-diffe","skin","axillar"]

colnames

#categorical data into numeric data

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

for value in ["sex","histologic-type","degree-of-diffe","skin","axillar"]:
    tumar_data[value].fillna(tumar_data[value].mode()[0],inplace = True)
    
tumar_data.head()

tumar_data.isnull().sum()


## Preproccessig of data using sklearn library/Converting cat data to numeric


from sklearn import preprocessing

le = preprocessing.LabelEncoder()

for x in colnames:
    tumar_data[x] =le.fit_transform(tumar_data[x])

tumar_data.head()

# 0---> <-1
# 1---> >-2

tumar_data.dtypes

#creating x and y
X = tumar_data.values[:,:-1]
Y = tumar_data.values[:,-1]

#standardized the x using scaler 

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

scaler.fit(X)

x =scaler.transform(X)
print(X)

Y = Y.astype(int)
Y

#Splitting the data in test and train

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3,random_state= 10)

X_train


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train,Y_train)

y_pred = model.predict(X_test)

print(list(zip(y_pred,Y_test))) 

#evalution of model


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

print(confusion_matrix(y_pred,Y_test))
print()
print(accuracy_score(y_pred,Y_test))
print()
print(classification_report(y_pred,Y_test))
print()

#tuning the model,adjusting the threshold


y_pred_prob = model.predict_proba(X_test)

y_pred_prob

#adjusting the threshold

y_pred_class=[]
for value in y_pred_prob[:,1]:
    if value > 0.4:
        y_pred_class.append(1)
    else:
        y_pred_class.append(0)
print (y_pred_class)

#evalution of model


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

print(confusion_matrix(y_pred_class,Y_test))
print()
print(accuracy_score(y_pred_class,Y_test))
print()
print(classification_report(y_pred_class,Y_test))
print()
