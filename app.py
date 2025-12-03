from flask import Flask, request, render_template
import pickle
import numpy as np
import time

app = Flask(__name__)

weather_classes = [
    'clear', 'cloudy', 'drizzly', 'foggy', 'hazey',
    'misty', 'rain', 'smokey', 'thunderstorm'
]

def load_model(model_path='model/model.pkl'):
    """Load trained ML model."""
    return pickle.load(open(model_path, 'rb'))

def classify_weather(features):
    """Run classification and return predicted label + latency."""
    model = load_model()
    start = time.time()
    
    prediction_index = model.predict(features)[0]
    
    latency = round((time.time() - start) * 1000, 2)

    # Corrected: return actual model prediction
    prediction = weather_classes[prediction_index]

    return prediction, latency


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Convert all fields to float — required for model.predict()
            temperature = float(request.form['temperature'])
            pressure = float(request.form['pressure'])
            humidity = float(request.form['humidity'])
            wind_speed = float(request.form['wind_speed'])
            wind_deg = float(request.form['wind_deg'])

            # Optional fields use .get() and default to 0
            rain_1h = float(request.form.get('rain_1h', 0) or 0)
            rain_3h = float(request.form.get('rain_3h', 0) or 0)
            snow = float(request.form.get('snow', 0) or 0)
            clouds = float(request.form.get('clouds', 0) or 0)

            features = np.array([
                temperature, pressure, humidity,
                wind_speed, wind_deg, rain_1h,
                rain_3h, snow, clouds
            ]).reshape(1, -1)

            prediction, latency = classify_weather(features)

            return render_template('result.html',
                                   prediction=prediction,
                                   latency=latency)

        except Exception as e:
            # Required for unit test: Return form page containing "Error"
            error_msg = f"Error: {e}"
            return render_template('form.html', error=error_msg)

    # GET request: show the form
    return render_template('form.html')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
