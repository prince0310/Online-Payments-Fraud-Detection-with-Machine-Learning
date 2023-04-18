# Online-Payments-Fraud-Detection-with-Machine-Learning

The introduction of online payment systems has helped a lot in the ease of payments. But, at the same time, it increased in payment frauds. Online payment frauds can happen with anyone using any payment system, especially while making payments using a credit card. That is why detecting online payment fraud is very important for credit card companies to ensure that the customers are not getting charged for the products and services they never paid. 

## Online Payments Fraud Detection with Machine Learning
To identify online payment fraud with machine learning, we need to train a machine learning model for classifying fraudulent and non-fraudulent payments. For this, we need a dataset containing information about online payment fraud, so that we can understand what type of transactions lead to fraud. For this task, I collected a dataset from Kaggle, which contains historical information about fraudulent transactions which can be used to detect fraud in online payments. Below are all the columns from the dataset I’m using here:

* <font color='DarkOrange'>step</font> : represents a unit of time where 1 step equals 1 hour
* <font color='Pink'>type</font> : type of online transaction
* <font color='Orange'>amount</font> : the amount of the transaction
* <font color='Yellow'>nameOrig</font> : customer starting the transaction
* <font color='Purple'>oldbalanceOrg</font> : balance before the transaction
* <font color='Green'>newbalanceOrig</font> : balance after the transaction
* <font color='Brown'>nameDest</font> : recipient of the transaction
* <font color='Tomato'>oldbalanceDest</font> : initial balance of recipient before the transaction
* <font color='DarkKhaki'>newbalanceDest</font> : the new balance of recipient after the transaction
* <font color='red'>isFraud</font> : fraud transaction

## Online Payments Fraud Detection using Python

importing the necessary Python libraries and the [dataset](https://www.kaggle.com/ealaxi/paysim1/download) we need for this task:
```
import pandas as pd   
import numpy as np   

import matplotlib.pyplot as plt   
import seaborn as sns   
```
Read and show Dataset 
```
data = pd.read_csv(
    '/home/ryzenrtx/Prince/Projects/Online Payments Fraud Detection with Machine Learning/PS_20174392719_1491204439457_log.csv')
data.head()
```
![data](https://user-images.githubusercontent.com/85225054/232678285-25faa09e-6200-4b10-a1c6-3caef70d50ed.png)

Now, let’s have a look at whether this dataset has any null values or not:
```
print(data.isnull().sum()) 

```
![null](https://user-images.githubusercontent.com/85225054/232678804-df543314-f3eb-4166-98ec-ac4295f58317.png)

So this dataset does not have any null values. Before moving forward, now, let’s have a look at the type of transaction mentioned in the dataset:

![explore](https://user-images.githubusercontent.com/85225054/232678894-3bd7a0df-1737-49f8-a810-092fee0e9d73.png)


```
type = data["type"].value_counts()
transactions = type.index
quantity = type.values

import plotly.express as px
figure = px.pie(data, 
             values=quantity, 
             names=transactions,hole = 0.5, 
             title="Distribution of Transaction Type")
figure.show()
```
![graph](https://user-images.githubusercontent.com/85225054/232679154-bdae9007-9576-45cf-bfe0-c069e8dd8dbb.png)

Now let’s have a look at the correlation between the features of the data with the isFraud column:
```
# Checking correlation
correlation = data.corr()
sns.heatmap(correlation, annot=True)    

```
![corr](https://user-images.githubusercontent.com/85225054/232679420-eed1f10a-a597-46e8-bc4e-3f131c842042.png)

Now let’s transform the categorical features into numerical. Here I will also transform the values of the isFraud column into No Fraud and Fraud labels to have a better understanding of the output:

```
data["type"] = data["type"].map({"CASH_OUT": 1, "PAYMENT": 2, 
                                 "CASH_IN": 3, "TRANSFER": 4,
                                 "DEBIT": 5})
data["isFraud"] = data["isFraud"].map({0: "No Fraud", 1: "Fraud"})

```

### Online Payments Fraud Detection Model

Now let’s train a classification model to classify fraud and non-fraud transactions. Before training the model, I will split the data into training and test sets:

```
# splitting the data
from sklearn.model_selection import train_test_split
x = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
y = np.array(data[["isFraud"]])

```

Now let’s train the online payments fraud detection model:

```
# training a machine learning model
from sklearn.tree import DecisionTreeClassifier
xtrain, xtest, ytrain, ytest = train_test_split(
    x, y, test_size=0.10, random_state=42)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))
```

Now let’s classify whether a transaction is a fraud or not by feeding about a transaction into the model:
```
# prediction
#features = [type, amount, oldbalanceOrg, newbalanceOrig]
features = np.array([[4, 9000.60, 9000.60, 0.0]])
print(model.predict(features))

```
Dump Model with the help of pickle 
```
import pickle
pickle.dump(mode|l, open("model.pkl", "wb"))

```
loading the model 
```                            
model = pickle.load(open("model.pkl", "rb"))
```
Refrence : The code in this project was inspired by the [article](https://thecleverprogrammer.com/author/amankharwal/.) Online Payments Fraud Detection with Machine Learning.
