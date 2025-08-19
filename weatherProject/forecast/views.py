from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
import requests
import pandas as pd
import numpy as np
import pytz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta
import os

# API Configuration
API_KEY = '577927295df77ba001f2a115f0011903'
BASE_URL = 'https://api.openweathermap.org/data/2.5/'

# ======================
# Utility Functions
# ======================

def get_current_weather(city):
    """Fetch current weather data from OpenWeather API."""
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    
    data = response.json()
    return {
        "city": data.get("name"),
        "current_temp": round(data["main"]["temp"]),
        "feels_like": round(data["main"]["feels_like"]),
        "temp_min": round(data["main"]["temp_min"]),
        "temp_max": round(data["main"]["temp_max"]),
        "humidity": round(data["main"]["humidity"]),
        "description": data["weather"][0]["description"],
        "country": data["sys"]["country"],
        "wind_gust_dir": data["wind"]["deg"],
        "pressure": data["main"]["pressure"],
        "Wind_Gust_Speed": data["wind"]["speed"],
        'clouds': data['clouds']['all'],
        'visibility': data['visibility'],
    }

def read_historical_data(filename):
    """Read and clean historical weather CSV."""
    df = pd.read_csv(filename)
    df = df.dropna().drop_duplicates()
    return df

def prepare_data(data):
    """Prepare classification data for RainTomorrow prediction."""
    le_wind = LabelEncoder()
    le_rain = LabelEncoder()
    
    data['WindGustDir'] = le_wind.fit_transform(data['WindGustDir'])
    data['RainTomorrow'] = le_rain.fit_transform(data['RainTomorrow'])
    
    X = data[['MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed', 'Humidity', 'Pressure', 'Temp']]
    y = data['RainTomorrow']
    return X, y, le_wind, le_rain

def train_rain_model(X, y):
    """Train random forest classifier for rain prediction."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Rain Model Accuracy:", accuracy_score(y_test, y_pred))
    return model

def prepare_regression_data(data, feature):
    """Prepare lagged data for regression."""
    X, y = [], []
    for i in range(len(data) - 1):
        X.append(data[feature].iloc[i])
        y.append(data[feature].iloc[i + 1])
    return np.array(X).reshape(-1, 1), np.array(y)

def train_regression_model(X, y):
    """Train random forest regressor."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def predict_future(model, current_value):
    """Predict next 5 future values."""
    predictions = [current_value]
    for _ in range(5):
        next_value = model.predict(np.array([[predictions[-1]]]))
        predictions.append(next_value[0])
    return predictions[1:]

def map_wind_direction(wind_deg):
    """Map degrees to compass direction."""
    compass_points = [
        ("N", 0, 11.25), ("NNE", 11.25, 33.75), ("NE", 33.75, 56.25),
        ("ENE", 56.25, 78.75), ("E", 78.75, 101.25), ("ESE", 101.25, 123.75),
        ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75), ("S", 168.75, 191.25),
        ("SSW", 191.25, 213.75), ("SW", 213.75, 236.25), ("WSW", 236.25, 258.75),
        ("W", 258.75, 281.25), ("WNW", 281.25, 303.75), ("NW", 303.75, 326.25),
        ("NNW", 326.25, 348.75), ("N", 348.75, 360)
    ]
    wind_deg = wind_deg % 360
    return next((point for point, start, end in compass_points if start <= wind_deg < end), "N")

# ======================
# Shared Weather Logic
# ======================

def get_weather_context(city):
    """Fetch current weather, train models, and prepare forecast context."""
    # Fetch current weather
    current_weather = get_current_weather(city)
    if not current_weather:
        return None, "Invalid city name or API error."

    # Read historical CSV
    # Always use absolute path to weather.csv in project root
    csv_path = os.path.abspath(os.path.join(settings.BASE_DIR, '..', 'weather.csv'))
    if not os.path.exists(csv_path):
        return None, "Historical data file not found."
    
    historical_data = read_historical_data(csv_path)

    # Train rain prediction model
    X, y, le_wind, le_rain = prepare_data(historical_data)
    rain_model = train_rain_model(X, y)

    # Map wind direction and encode
    compass_direction = map_wind_direction(current_weather['wind_gust_dir'])
    if compass_direction in le_wind.classes_:
        compass_direction_encoded = le_wind.transform([compass_direction])[0]
    else:
        compass_direction_encoded = -1

    # Prepare current weather data for rain prediction
    current_df = pd.DataFrame([{
        "MinTemp": current_weather['temp_min'],
        "MaxTemp": current_weather['temp_max'],
        "WindGustDir": compass_direction_encoded,
        "WindGustSpeed": current_weather['Wind_Gust_Speed'],
        "Humidity": current_weather['humidity'],
        "Pressure": current_weather['pressure'],
        "Temp": current_weather['current_temp']
    }])
    rain_prediction = rain_model.predict(current_df)[0]

    # Train regression models for temp & humidity
    X_temp, y_temp = prepare_regression_data(historical_data, 'Temp')
    X_hum, y_hum = prepare_regression_data(historical_data, 'Humidity')
    temp_model = train_regression_model(X_temp, y_temp)
    hum_model = train_regression_model(X_hum, y_hum)

    # Predict next 5 hours
    future_temp = predict_future(temp_model, current_weather['current_temp'])
    future_humidity = predict_future(hum_model, current_weather['humidity'])

    # Prepare future time labels
    timezone = pytz.timezone("Asia/Karachi")
    now = datetime.now(timezone)
    next_hour = now + timedelta(hours=1)
    next_hour = next_hour.replace(minute=0, second=0, microsecond=0)
    future_times = [(next_hour + timedelta(hours=i)).strftime("%H:00") for i in range(len(future_temp))]

    # Build context
    context = {
        'city': current_weather['city'],
        'country': current_weather['country'],
        'current_temp': current_weather['current_temp'],
        'feels_like': current_weather['feels_like'],
        'temp_min': current_weather['temp_min'],
        'temp_max': current_weather['temp_max'],
        'humidity': current_weather['humidity'],
        'description': current_weather['description'],
        'wind': current_weather['wind_gust_dir'],
        'pressure': current_weather['pressure'],
        'Wind_Gust_Speed': current_weather['Wind_Gust_Speed'],
        'clouds': current_weather['clouds'],
        'visibility': current_weather['visibility'],
        'time': now.strftime("%H:%M"),
        'date': now.strftime("%d/%m/%Y"),
        'rain_prediction': 'Yes' if rain_prediction else 'No',
        'future_forecast': list(zip(future_times, [round(t, 1) for t in future_temp], [round(h, 1) for h in future_humidity]))
    }
    return context, None

# ========================
# Django View
# ========================

def weather_view(request):
    city = request.POST.get('city') if request.method == 'POST' else 'Allahabad'
    context, error = get_weather_context(city)
    if error:
        # Pass error message to template for display
        context = {'error_message': 'Enter a valid city name.'}
        # Preserve entered city in input field
        if request.method == 'POST':
            context['city'] = city
    return render(request, 'forecast/weather.html', context)
