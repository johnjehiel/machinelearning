from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# load the pickle model or source the code for the model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split #function for training and testing
from sklearn.feature_extraction.text import TfidfVectorizer #text to numerical data
from sklearn.linear_model import LogisticRegression #the algorithm that we use
from sklearn.metrics import accuracy_score #to evaluate our model

raw_mail_data = pd.read_csv("D:\workspace python\mlprojects\spam prediction project\mail_data.csv")
mail_data = raw_mail_data.where(pd.notnull(raw_mail_data),"")
mail_data.loc[mail_data["Category"]=='spam',"Category",] = 0
mail_data.loc[mail_data["Category"]=='ham',"Category",] = 1
x = mail_data['Message']
y = mail_data['Category']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)
feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase=True) #lowercase can take True or 1
x_train_features = feature_extraction.fit_transform(x_train) # ERROR here in vscode :(

x_test_features = feature_extraction.transform(x_test) 
y_train = y_train.astype('int')
y_test = y_test.astype('int')
model = LogisticRegression()
model.fit(x_train_features,y_train)


@app.route('/')
def Home():
    return render_template('index.html')

@app.route('/predict', methods = ["POST"])
def predict():
    mail = [request.form.get("rawmail")]
    mail = feature_extraction.transform(mail)
    if (model.predict(mail)==1):
        prediction = "HAM"
    else:
        prediction = "SPAM"
    #prediction_text is used in index.html using a jinja
    return render_template("index.html", prediction_text=f"The mail is a {prediction}")




if __name__=="__main__":
    app.run(debug=True)