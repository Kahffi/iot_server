from flask import Flask
import pickle
import numpy as np 


model = pickle.load(open("../knn_model.pkl", "rb"))
app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello, World!'

@app.route('/about')
def about():
    return 'About'

@app.route('/predict')# gets the values that were sent to '/predict' by 'index.html'
def predict():
    final_features = [np.array([1.0,0.02394, 523.54])]# turns the form values into a Numpy array
    prediction = model.predict(final_features)# makes a prediction using the values in the created Numpy array

    output = prediction[0]# get the prediction as a string

    return output# displays the prediction inside the '<b>{{ prediction_text }}</b>' that we've seen in 'index.html'