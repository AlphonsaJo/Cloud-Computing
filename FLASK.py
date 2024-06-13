from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained Linear Regression model
lr_model = joblib.load('lr_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    data = request.form.to_dict()
    cpu_usage = float(data['cpu_usage'])
    memory_usage = float(data['memory_usage'])
    network_traffic = float(data['network_traffic'])
    power_consumption = float(data['power_consumption'])
    num_executed_instructions = float(data['num_executed_instructions'])
    execution_time = float(data['execution_time'])

    # Prepare the input data as a NumPy array
    input_data = np.array([[cpu_usage, memory_usage, network_traffic, power_consumption, num_executed_instructions, execution_time]])

    # Make prediction using the loaded model
    prediction = lr_model.predict(input_data)

    # Return the prediction as JSON response
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
