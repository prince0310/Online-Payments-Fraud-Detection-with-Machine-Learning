from keras.models import load_model
import numpy as np


from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('model.h5')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    open = float(request.form['open'])
    high = float(request.form['high'])
    low = float(request.form['low'])
    volume = float(request.form['volume'])

    # Create input array for prediction
    input_array = np.array([[open, high, low, volume]])

    # Make prediction using the loaded model
    prediction = model.predict(input_array)

    # Extract the predicted output value
    output = prediction[0][0]

    return render_template('index.html', prediction=output)


if __name__ == '__main__':
    app.run(debug=True)
