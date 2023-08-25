from flask import Flask, request, render_template
import pickle
import numpy as np
import sklearn

# load the model from the pickle file
model = pickle.load(open("placement_model_v1.pkl", "rb"))

# create the flask app
app = Flask(__name__)


# define the home page
@app.route("/")
def home():
    return render_template("home1.html")


# define the prediction page
@app.route("/predict", methods=["POST"])
def predict():
    # get the features from the form
    features = [float(x) for x in request.form.values()]
    # convert the features to a numpy array
    features = np.array(features).reshape(1, -1)
    # make the prediction using the model
    prediction = model.predict(features)
    # convert the prediction to a string
    if prediction == 0:
        output = "Not Placed"
    else:
        output = "Placed"
    # return the output to the user
    return render_template("home1.html", prediction_text="The predicted output is {}".format(output))


# run the app
if __name__ == "__main__":
    app.run(debug=True)
