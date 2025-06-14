from flask import request, Flask, render_template
import jsonify
import joblib

app = Flask(__name__)
app.app_context().push()

@app.route("/")
def home():
    ''' Render the home page with labels for the input fields '''

    # Define the labels for the input fields
    # These labels correspond to the features used in the model
    # 'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'
    l = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
    return render_template("index.html", labels=l)

@app.route("/api/data", methods=["POST"])
def post_data():
    ''' Endpoint to receive data and return prediction '''

    # Check if the request contains JSON data
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    # Get the JSON data from the request
    # and print it for debugging
    data = request.get_json()
    print(data)
    
    # Load the model
    model = joblib.load('model.pkl')
    
    # Prepare the input data
    input_data = [[data['MedInc'], data['HouseAge'], data['AveRooms'], 
                   data['AveBedrms'], data['Population'], data['AveOccup'], 
                   data['Latitude'], data['Longitude']]]
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Return the prediction as JSON
    return jsonify({"prediction": prediction[0]*100000}), 200  # Scale the prediction to match the original target variable's scale

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)  # Run the Flask app on all interfaces at port 5000