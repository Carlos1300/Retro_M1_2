"""

Author: Carlos de Jesús Ávila González
Title: Decision Tree Classifier Using Scikit-Learn
Date: 12/09/2022

"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier     #Importing the model from the framework.
from sklearn.model_selection import train_test_split    #Importing the splitter.
from sklearn import metrics     #Importing the metrics for our model.

columns = ["pregnant", "glucose", "bp", "skin", "insulin", "bmi", "pedigree", "age", "label"]   #Name of the columns to be used.

df = pd.read_csv("Modulo_2\pima-indians-diabetes.csv", header=None, names=columns)  #Importing the data we are going to use.

features = ["pregnant", "glucose", "bp", "skin", "insulin", "bmi", "pedigree", "age"]   #Name of the feature columns we are going to use.

X = df[features]    #Features to use on our model.
x = df[features].values     #Values of the features.
y = df.label    #Targets of our model.

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, 
                                                    random_state=40)    #Splitting the data

print('-------------------DECISION TREE-------------------')

tree_clf = DecisionTreeClassifier()     #Define our model.
tree_clf = tree_clf.fit(X_train,y_train)    #Fit our model to our data.

train_pred = tree_clf.predict(X_train)

y_pred = tree_clf.predict(X_test)   #Predict with out model.

accuracy = round(metrics.accuracy_score(y_train, train_pred),2)*100  #Calculate the accuracy of the model.
print("The accuracy of the model in train is: %s" %(accuracy), "%")
cmat = metrics.confusion_matrix(y_train, train_pred)     #Obtain the confusion matrix of the model.
print("Confusion Matrix of the model in train:")
print(cmat)

accuracy = round(metrics.accuracy_score(y_test, y_pred),2)*100  #Calculate the accuracy of the model.
print("The accuracy of the model in test is: %s" %(accuracy), "%")
cmat = metrics.confusion_matrix(y_test, y_pred)     #Obtain the confusion matrix of the model.
print("Confusion Matrix of the model in test:")
print(cmat)

print('----------------------------------------------------')


################ PREDICTIONS ########################


for i in range(10):

    #Make some data so we can predict
    
    predict_data = [[np.random.randint(np.min(X["pregnant"]), np.max(X["pregnant"])), 
                    np.random.randint(np.min(X["glucose"]), np.max(X["glucose"])),
                    np.random.randint(np.min(X["bp"]), np.max(X["bp"])),
                    np.random.randint(np.min(X["skin"]), np.max(X["skin"])),
                    np.random.randint(np.min(X["insulin"]), np.max(X["insulin"])),
                    np.random.choice(X["bmi"]),
                    np.random.choice(X["pedigree"]),
                    np.random.randint(np.min(X["age"]), np.max(X["age"]))]]

    print('\n', '-'*25)
    print("Prediction for the sample: %s\t Class: %s" %(predict_data, tree_clf.predict(predict_data)))
    print("Probability of class for the sample: %s\t Probability: %s" %(predict_data, tree_clf.predict_proba(predict_data)))
    print('-'*25, '\n')


################ TUNED DECISION TREE ########################

print('-------------------TUNED DECISION TREE-------------------')

tuned_tree_clf = DecisionTreeClassifier(criterion='gini', max_depth=3)  #Obtain our tuned model.
tuned_tree_clf = tuned_tree_clf.fit(X_train,y_train)

train_pred = tuned_tree_clf.predict(X_train)

y_pred_tuned = tuned_tree_clf.predict(X_test)

accuracy = round(metrics.accuracy_score(y_train, train_pred),2)*100  #Calculate the accuracy of the model.
print("The accuracy of the model in train is: %s" %(accuracy), "%")
cmat = metrics.confusion_matrix(y_train, train_pred)     #Obtain the confusion matrix of the model.
print("Confusion Matrix of the model in train:")
print(cmat)

accuracy = round(metrics.accuracy_score(y_test, y_pred_tuned),2)*100  #Calculate the accuracy of the model.
print("The accuracy of the model in test is: %s" %(accuracy), "%")
cmat = metrics.confusion_matrix(y_test, y_pred_tuned)     #Obtain the confusion matrix of the model.
print("Confusion Matrix of the model in test:")
print(cmat)

print('----------------------------------------------------')

################ TUNED PREDICTIONS ########################


for i in range(10):

    predict_data = [[np.random.randint(np.min(X["pregnant"]), np.max(X["pregnant"])), 
                    np.random.randint(np.min(X["glucose"]), np.max(X["glucose"])),
                    np.random.randint(np.min(X["bp"]), np.max(X["bp"])),
                    np.random.randint(np.min(X["skin"]), np.max(X["skin"])),
                    np.random.randint(np.min(X["insulin"]), np.max(X["insulin"])),
                    np.random.choice(X["bmi"]),
                    np.random.choice(X["pedigree"]),
                    np.random.randint(np.min(X["age"]), np.max(X["age"]))]]

    print('\n', '-'*25)
    print("Prediction for the sample with tuned tree: %s\t Class: %s" %(predict_data, tuned_tree_clf.predict(predict_data)))
    print("Probability of class for the sample with tuned tree: %s\t Probability: %s" %(predict_data, tuned_tree_clf.predict_proba(predict_data)))
    print('-'*25, '\n')