import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split #function for training and testing
from sklearn.feature_extraction.text import TfidfVectorizer #text to numerical data
from sklearn.linear_model import LogisticRegression #the algorithm that we use
from sklearn.metrics import accuracy_score #to evaluate our model

import pickle

raw_mail_data = pd.read_csv("D:\workspace python\mlprojects\spam prediction project\mail_data.csv")
#print(raw_mail_data.head(5))
mail_data = raw_mail_data.where(pd.notnull(raw_mail_data),"")
mail_data.loc[mail_data["Category"]=='spam',"Category",] = 0
mail_data.loc[mail_data["Category"]=='ham',"Category",] = 1
x = mail_data['Message']
y = mail_data['Category']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)
feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase='True')
x_train_features = feature_extraction.fit_transform(x_train) # ERROR here in vscode :(

x_test_features = feature_extraction.transform(x_test) 
y_train = y_train.astype('int')
y_test = y_test.astype('int')
model = LogisticRegression()
model.fit(x_train_features,y_train)


#mail = ["Here is your discount code RP176781. To stop further messages reply stop. www.regalportfolio.co.uk. Customer Services 08717205546"]
mail = [input()]
#text to feature vectors
#feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase='True')
mail = feature_extraction.transform(mail)

if (model.predict(mail)==1):
    print("HAM")
else:
    print("SPAM")

'''

rawmail = ["Here is your discount code RP176781. To stop further messages reply stop. www.regalportfolio.co.uk. Customer Services 08717205546"]
with open("feature_extraction_SM",'rb') as f:
    fe = pickle.load(f)
    
mail = fe.transform(rawmail)
with open("spam_mail_prediction_model","rb") as f:
    mp = pickle.load(f)
if (mp.predict(mail)==1):
    print("HAM")
else:
    print("SPAM")
'''