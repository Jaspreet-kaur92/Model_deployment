 

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Creating flask app
app = Flask(__name__)

# Loading the pickle file
model = pickle.load(open('model.pkl','rb'))

# Creating a home page (/) bydefault this will render_template('index.html') page.
@app.route('/')
def home():
    return render_template('index.html')

# Creating /predict which is basically a POST method
# Wherein we are provinding some features of our model.pkl file
# So that our model will take those inputs and give us some output.
# '/predict' this is our web api
@app.route('/predict', methods=['POST'])
def predict():
    ## For rendering results on html GUI
    # request.form.values() -> It will take input from all the forms
    # we are using request library to take all the values from the text field 
    # and store it in int_feature column  
    int_feature = [int(x) for x in request.form.values()]
    # Converting these features into array
    final_features = [np.array(int_feature)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    # this prediction_text will get replace with {{ prediction_text }} in index.html page.
    return render_template('index.html', prediction_text = 'Employee salary sholud be $ {}'.format(output))

# This is main function which will run this whole flask
if __name__=='__main__':
    app.run(debug=True)
