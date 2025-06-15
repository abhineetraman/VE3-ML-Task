from flask import request, Flask, render_template
from flask import jsonify
import joblib
import pandas as pd

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
    
    keys_map = {
        'MedInc': data.get('MedInc', data.get('medinc')),
        'HouseAge': data.get('HouseAge', data.get('houseage')),
        'AveRooms': data.get('AveRooms', data.get('averooms')),
        'AveBedrms': data.get('AveBedrms', data.get('avebedrms')),
        'Population': data.get('Population', data.get('population')),
        'AveOccup': data.get('AveOccup', data.get('aveoccup')),
        'Latitude': data.get('Latitude', data.get('latitude')),
        'Longitude': data.get('Longitude', data.get('longitude')),
    }
    # Create DataFrame with correct column names
    input_df = pd.DataFrame([{
        'MedInc': float(keys_map['MedInc']),
        'HouseAge': float(keys_map['HouseAge']),
        'AveRooms': float(keys_map['AveRooms']),
        'AveBedrms': float(keys_map['AveBedrms']),
        'Population': float(keys_map['Population']),
        'AveOccup': float(keys_map['AveOccup']),
        'Latitude': float(keys_map['Latitude']),
        'Longitude': float(keys_map['Longitude'])
    }])
    prediction = model.predict(input_df)[0]
    print(f"Prediction: {prediction}")
    return jsonify({'prediction': round(float(prediction)*100000,2)})


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)  # Run the Flask app on all interfaces at port 5000