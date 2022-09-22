import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app = Flask(__name__) # Starting point of flask
## Load the model
regmodel=pickle.load(open('regmodel.pkl','rb')) # open the pickle file and load it
scalar=pickle.load(open('scaling.pkl','rb'))

@app.route('/') # First root or home page
def home():
    return render_template('home.html') # It will return the html page, home.html is not yet defined

@app.route('/predict_api',methods=['POST'])  # create a predict api to send a request to app using POST

def predict_api():   
    data=request.json['data'] # whenever you hit predict api with information you need to capture it using request.json and stores in data
    print(data) # It is in json format
    print(np.array(list(data.values())).reshape(1,-1)) # Get the values and convert into list for single values and transform the data into single line
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST']) ## New method predict for web form insted of api
def predict():
    data=[float(x) for x  in request.form.values()] # Convert all the values to float (as in dataset)
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=regmodel.predict(final_input)[0] # Regression model for prediction, output will be two dimensional hence pick the 1st dimension
    return render_template("home.html",prediction_text="House price predicted is {}".format(output))


## To run
if __name__=="__main__":
    app.run(debug=True)
