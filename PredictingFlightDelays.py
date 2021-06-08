#!/usr/bin/env python
# coding: utf-8

# # Deep Learning to Predict Delayed Flights in Tableau 

# In[2]:



import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict, train_test_split 
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing, metrics, svm, ensemble
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, confusion_matrix, fbeta_score

import keras
import pickle
from keras_pickle_wrapper import KerasPickleWrapper 
import tabpy
import tabpy_client
from tabpy.tabpy_tools.client import Client


# In[3]:


import warnings
warnings.filterwarnings('ignore')


# ## Load Data

# In[4]:


# load data
df = pd.read_csv('flightdata.csv')
df.head()


# ## EDA

# In[5]:


# inspect cancelled flights
df[df['CANCELLED'] != 0.0]


# In[6]:


# inspect diverted flights
df[df['DIVERTED'] != 0.0]


# In[7]:


df.shape


# In[8]:


df.isnull().values.any()


# In[9]:


df.isnull().sum()


# In[10]:


df = df.drop('Unnamed: 25', axis=1)


# In[11]:


df[df.isnull().values.any(axis=1)]


# ## Feature Engineering

# In[12]:


# fill in NAs with 1 as INT
df = df.fillna({'ARR_DEL15': 1})
df['ARR_DEL15'] = df['ARR_DEL15'].astype(int)
df.iloc[177:185]


# In[13]:


df.head()


# In[14]:


# bin the times of day hourly
for index, row in df.iterrows():
    df.loc[index, 'CRS_DEP_TIME'] = math.floor(row['CRS_DEP_TIME'] / 100)
df.head()


# ## Feature Selection and Pre Processing

# In[15]:


# save data to upload to tableau
df.to_csv('flight_delays_clean.csv')


# In[16]:


# select feeatures
delay_df = df[["MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "ORIGIN", "DEST", "CRS_DEP_TIME", "ARR_DEL15"]]
delay_df.isnull().sum()


# In[17]:


# get dummies for machine learning algorithms
delay_df = pd.get_dummies(delay_df, columns=['ORIGIN', 'DEST'])
delay_df.head()


# In[18]:


# subset the entire data for deep learning deployment
deep_df = delay_df.copy()
x = deep_df.drop('ARR_DEL15', axis=1)
y = deep_df['ARR_DEL15']


# ## Split and Scale

# In[19]:


# stratify split to better handle class imbalances
x_train, x_test, y_train, y_test = train_test_split(delay_df.drop('ARR_DEL15', axis=1), delay_df['ARR_DEL15'], 
                                                    test_size=0.20, stratify=delay_df['ARR_DEL15'], random_state=42)


# In[20]:


Counter(y_train)


# In[21]:


Counter(y_test)


# In[22]:


# weight each class for class imbalances in deep learning
counts = np.bincount(y_train)
print(
    "Number of positive samples in training data: {} ({:.2f}% of total)".format(
        counts[1], 100 * float(counts[1]) / len(y_train)
    )
)

weight_for_0 = 1.0 / counts[0]
weight_for_1 = 1.0 / counts[1]


# In[23]:


weight_for_0 


# In[24]:


weight_for_1


# In[25]:


# scale the data
#scaler = preprocessing.StandardScaler().fit(x_train)
#scaler = preprocessing.MinMaxScaler().fit(x_train)
#x_train = scaler.transform(x_train)


# ## Random Forest Classifier

# In[26]:


rfc = RandomForestClassifier(random_state=42)
rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_test)


# In[27]:


# Show classification report for the best model (set of parameters) run over the full dataset
print("Classification report:")
print(classification_report(y_test, y_pred))


# In[28]:


# Show accuracy and area under ROC curve
print("Accuracy: %0.3f" % accuracy_score(y_test, y_pred, normalize=True))
print("Aucroc: %0.3f" % metrics.roc_auc_score(y_test, y_pred))
print("")


# In[29]:


cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(cm,
                     index = ['ontime','delayed'], 
                     columns = ['ontime','delayed'])

plt.figure(figsize=(5.5,4))
sns.heatmap(cm_df, annot=True, cmap='Spectral', fmt='d')
plt.title('Random Forest Classifier', fontsize=20)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

#extracting true_positives, false_positives, true_negatives, false_negatives
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("True Negatives: ",tn)
print("False Positives: ",fp)
print("False Negatives: ",fn)
print("True Positives: ",tp)


# ## Logistic Regression

# In[30]:


# Logistic regression with 10 fold stratified cross-validation using model specific cross-validation in scikit-learn

lgclf = LogisticRegressionCV(Cs=list(np.power(10.0, np.arange(-10, 10))),penalty='l2',scoring='roc_auc',cv=10)

lgclf.fit(x_train, y_train)

y_pred = lgclf.predict(x_test)


# In[31]:


# Show classification report 
print("Classification report:")
print(classification_report(y_test, y_pred))


# In[32]:



# Show accuracy and area under ROC curve
print("Accuracy: %0.3f" % accuracy_score(y_test, y_pred, normalize=True))
print("Aucroc: %0.3f" % metrics.roc_auc_score(y_test, y_pred))


# In[33]:


cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(cm,
                     index = ['ontime','delayed'], 
                     columns = ['ontime','delayed'])

plt.figure(figsize=(5.5,4))
sns.heatmap(cm_df, annot=True, cmap='Spectral', fmt='d')
plt.title('Logisitic Regression', fontsize=20)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

#extracting true_positives, false_positives, true_negatives, false_negatives
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("True Negatives: ",tn)
print("False Positives: ",fp)
print("False Negatives: ",fn)
print("True Positives: ",tp)


# ## Naive Bayes

# In[34]:


# Naive Bayes with 10 fold stratified cross-validation
nbclf = GaussianNB()
scores = cross_val_score(nbclf, x_train, y_train, cv=10, scoring='roc_auc')

# Show accuracy statistics for cross-validation
print("Accuracy: %0.3f" % (scores.mean()))
print("Aucroc: %0.3f" % metrics.roc_auc_score(y_test, cross_val_predict(nbclf, x_test, y_test, cv=10)))


# In[35]:


cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(cm,
                     index = ['ontime','delayed'], 
                     columns = ['ontime','delayed'])

plt.figure(figsize=(5.5,4))
sns.heatmap(cm_df, annot=True, cmap='Spectral', fmt='d')
plt.title('Naive Bayes', fontsize=20)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

#extracting true_positives, false_positives, true_negatives, false_negatives
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("True Negatives: ",tn)
print("False Positives: ",fp)
print("False Negatives: ",fn)
print("True Positives: ",tp)


# ## Gradient Boosting Classifier

# In[36]:


# Define the parameter grid to use for tuning the Gradient Boosting Classifier
gridparams = dict(learning_rate=[0.01, 0.1],loss=['deviance','exponential'])

# Parameters we're not tuning for this classifier
params = {'n_estimators': 1500, 'max_depth': 4}

# Setup for grid search with cross-validation for Gradient Boosting Classifier
# n_jobs=-1 for parallel execution using all available cores
gbclf = GridSearchCV(ensemble.GradientBoostingClassifier(**params), gridparams, cv=10, scoring='roc_auc',n_jobs=-1)
gbclf.fit(x_train, y_train)

# Show the definition of the best model
print("Best model:")
print(gbclf.best_estimator_)
print("")

# Show classification report for the best model (set of parameters) run over the full dataset
print("Classification report:")    
y_pred = gbclf.predict(x_test)
print(classification_report(y_test, y_pred))

# Show accuracy and area under ROC curve
print("Accuracy: %0.3f" % accuracy_score(y_test, y_pred, normalize=True))
print("Aucroc: %0.3f" % metrics.roc_auc_score(y_test, y_pred))


# In[37]:


cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(cm,
                     index = ['ontime','delayed'], 
                     columns = ['ontime','delayed'])

plt.figure(figsize=(5.5,4))
sns.heatmap(cm_df, annot=True, cmap='Spectral', fmt='d')
plt.title('Gradient Boosting Classifier', fontsize=20)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

#extracting true_positives, false_positives, true_negatives, false_negatives
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("True Negatives: ",tn)
print("False Positives: ",fp)
print("False Negatives: ",fn)
print("True Positives: ",tp)


# ## Keras Deep Learning - Train and Test

# In[38]:


test_model = keras.Sequential(
    [
        keras.layers.Dense(
            256, activation="relu", input_shape=(x_train.shape[-1],)
        ),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.01),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)
test_model.summary()


# In[39]:


def FindBatchSize(model):
    """model: model architecture, that is yet to be trained"""
    import os, sys, psutil, gc, tensorflow, keras
    import numpy as np
    from keras import backend as K
    BatchFound= 16

    try:
        total_params= int(test_model.count_params());    GCPU= "CPU"
        #find whether gpu is available
        try:
            if K.tensorflow_backend._get_available_gpus()== []:
                GCPU= "CPU";    #CPU and Cuda9GPU
            else:
                GCPU= "GPU"
        except:
            from tensorflow.python.client import device_lib;    #Cuda8GPU
            def get_available_gpus():
                local_device_protos= device_lib.list_local_devices()
                return [x.name for x in local_device_protos if x.device_type == 'GPU']
            if "gpu" not in str(get_available_gpus()).lower():
                GCPU= "CPU"
            else:
                GCPU= "GPU"

        #decide batch size on the basis of GPU availability and model complexity
        if (GCPU== "GPU") and (os.cpu_count() >15) and (total_params <1000000):
            BatchFound= 64    
        if (os.cpu_count() <16) and (total_params <500000):
            BatchFound= 64  
        if (GCPU== "GPU") and (os.cpu_count() >15) and (total_params <2000000) and (total_params >=1000000):
            BatchFound= 32      
        if (GCPU== "GPU") and (os.cpu_count() >15) and (total_params >=2000000) and (total_params <10000000):
            BatchFound= 16  
        if (GCPU== "GPU") and (os.cpu_count() >15) and (total_params >=10000000):
            BatchFound= 8       
        if (os.cpu_count() <16) and (total_params >5000000):
            BatchFound= 8    
        if total_params >100000000:
            BatchFound= 1

    except:
        pass
    try:

        #find percentage of memory used
        memoryused= psutil.virtual_memory()
        memoryused= float(str(memoryused).replace(" ", "").split("percent=")[1].split(",")[0])
        if memoryused >75.0:
            BatchFound= 8
        if memoryused >85.0:
            BatchFound= 4
        if memoryused >90.0:
            BatchFound= 2
        if total_params >100000000:
            BatchFound= 1
        print("Batch Size:  "+ str(BatchFound));    gc.collect()
    except:
        pass

    memoryused= [];    total_params= [];    GCPU= "";
    del memoryused, total_params, GCPU;    gc.collect()
    return BatchFound


FindBatchSize(test_model)


# In[40]:


metrics = [
    keras.metrics.BinaryAccuracy(name="accuracy"),
    keras.metrics.FalseNegatives(name="fn"),
    keras.metrics.FalsePositives(name="fp"),
    keras.metrics.TrueNegatives(name="tn"),
    keras.metrics.TruePositives(name="tp"),
    keras.metrics.Precision(name="precision"),
    keras.metrics.Recall(name="recall"),
    keras.metrics.AUC(name="auc"),
]

test_model.compile(
    optimizer=keras.optimizers.Adam(1e-2), loss="binary_crossentropy", metrics=metrics
)

callbacks = [keras.callbacks.ModelCheckpoint("delay_test_at_epoch_{epoch}.h5")]
class_weight = {0: weight_for_0, 1: weight_for_1}

test_model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=100,
    verbose=0,
    callbacks=callbacks,
    validation_split = 0.2,
   class_weight=class_weight,
)


# In[41]:



# Evaluate your model accuracy on the test set

accuracy_score = test_model.evaluate(x_test, y_test)

print("Binary Crossentropy: %0.3f" % (accuracy_score[0]))
print("Accuracy: %0.3f" % (accuracy_score[1]))
print("Precision: %0.3f" % (accuracy_score[6]))
print("Recall: %0.3f" % (accuracy_score[7]))
print("AUC: %0.3f" % (accuracy_score[8]))


# In[56]:


cm = np.array([[accuracy_score[4], accuracy_score[3]], [accuracy_score[2], accuracy_score[5]]])

cm_df = pd.DataFrame(cm,
                     index = ['ontime','delayed'], 
                     columns = ['ontime','delayed'])

plt.figure(figsize=(5.5,4))
sns.heatmap(cm_df, annot=True, cmap='Spectral', fmt='g')
plt.title('Kears Deep Learning Classifier', fontsize=20)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

#extracting true_positives, false_positives, true_negatives, false_negatives
print("True Negatives: ",int(accuracy_score[4]))
print("False Positives: ",int(accuracy_score[3]))
print("False Negatives: ",int(accuracy_score[2]))
print("True Positives: ",int(accuracy_score[5]))


# In[43]:


keras.backend.clear_session()


# In[ ]:





# ## Keras Deep Learning - Full Model

# In[44]:


# weaight each class for class imbalances
counts = np.bincount(y)
print(
    "Number of positive samples in training data: {} ({:.2f}% of total)".format(
        counts[1], 100 * float(counts[1]) / len(y)
    )
)

weight_for_0 = 1.0 / counts[0]
weight_for_1 = 1.0 / counts[1]


# In[45]:


model = keras.Sequential(
    [
        keras.layers.Dense(
            256, activation="relu", input_shape=(x.shape[-1],)
        ),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.01),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)
model.summary()


# In[46]:



model.compile(
    optimizer=keras.optimizers.Adam(1e-2), loss="binary_crossentropy"
)

callbacks = [keras.callbacks.ModelCheckpoint("delay_model_at_epoch_{epoch}.h5")]
class_weight = {0: weight_for_0, 1: weight_for_1}

kpw = KerasPickleWrapper(model)

kpw().fit(
    x,
    y,
    batch_size=64,
    epochs=100,
    verbose=0,
    callbacks=callbacks,
    validation_split = 0.2,
   class_weight=class_weight,
)


# ## Deploy Model to Tableau with TabPy

# In[47]:


client = tabpy_client.Client('http://localhost:9004/')


# In[48]:



def PredictDelay(_arg1, _arg2, _arg3, _arg4, _arg5, _arg6):
    
    import pandas as pd
    
    #handle dummy variable assignments
    ORIGIN_ATL = 1 if _arg4[0] == 'ATL' else 0
    ORIGIN_DTW = 1 if _arg4[0] == 'DTW' else 0
    ORIGIN_JFK = 1 if _arg4[0] == 'JFK' else 0
    ORIGIN_MSP = 1 if _arg4[0] == 'MSP' else 0
    ORIGIN_SEA = 1 if _arg4[0] == 'SEA' else 0

    DEST_ATL = 1 if _arg5[0] == 'ATL' else 0
    DEST_DTW = 1 if _arg5[0] == 'DTW' else 0
    DEST_JFK = 1 if _arg5[0] == 'JFK' else 0
    DEST_MSP = 1 if _arg5[0] == 'MSP' else 0
    DEST_SEA = 1 if _arg5[0] == 'SEA' else 0
    
    # create a data dictionary
    row = {'MONTH': _arg1,
           'DAY_OF_MONTH': _arg2,
           'DAY_OF_WEEK': _arg3,
           'CRS_DEP_TIME': _arg6,
           
           'ORIGIN_ATL': ORIGIN_ATL,
           'ORIGIN_DTW': ORIGIN_DTW,
           'ORIGIN_JFK': ORIGIN_JFK,
           'ORIGIN_MSP': ORIGIN_MSP,
           'ORIGIN_SEA': ORIGIN_SEA,
           
           'DEST_ATL': DEST_ATL,
           'DEST_DTW': DEST_DTW,
           'DEST_JFK': DEST_JFK,
           'DEST_MSP': DEST_MSP,
           'DEST_SEA': DEST_SEA,
           }

    # convert it into a dataframe
    X = pd.DataFrame(data = row, index=[0]) 


    # return prediction as a string since float32 cannot be serialized
    return str(kpw().predict(X)[0][0])    


# In[49]:


# Publish the PredictDelay function to TabPy server so it can be used from Tableau
client.deploy('PredictDelay',
                  PredictDelay,
                  'Returns the probability of a delayed flight', override = True)


# In[ ]:





# In[50]:


# testing
kpw().predict(np.column_stack([5, 5, 5, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]))[0][0]


# In[ ]:





# In[51]:


keras.backend.clear_session()


# In[ ]:





# In[ ]:




