from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
app.config["DEBUG"] = True

#Home page route - no parameters and using render_template to display the welcome.html file
@app.route("/")
def home():
    return render_template("welcome.html")

#Body parameter route
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        model = pickle.load(open("food_delivery_model.pkl", "rb"))
        
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form

        distance_km = data.get('Distance_km')
        weather_clear = data.get('Weather_Clear')
        weather_foggy = data.get('Weather_Foggy')
        weather_rainy = data.get('Weather_Rainy')
        weather_snowy = data.get('Weather_Snowy')
        weather_windy = data.get('Weather_Windy')
        traffic_level_low = data.get('Traffic_Level_Low')
        traffic_level_medium = data.get('Traffic_Level_Medium')
        traffic_level_high = data.get('Traffic_Level_High')
        time_of_day_afternoon = data.get('Time_of_Day_Afternoon')
        time_of_day_evening = data.get('Time_of_Day_Evening')
        time_of_day_morning = data.get('Time_of_Day_Morning')
        time_of_day_night = data.get('Time_of_Day_Night')
        vehicle_type_bike = data.get('Vehicle_Type_Bike')
        vehicle_type_car = data.get('Vehicle_Type_Car')
        vehicle_type_scooter = data.get('Vehicle_Type_Scooter')
        preparation_time_min = data.get('Preparation_Time_min')
        courier_experience_yrs = data.get('Courier_Experience_yrs')

        input_data = {
            'Distance_km': [distance_km],
            'Weather_Clear': [weather_clear],
            'Weather_Foggy': [weather_foggy],
            'Weather_Rainy': [weather_rainy],
            'Weather_Snowy': [weather_snowy],
            'Weather_Windy': [weather_windy],
            'Traffic_Level_Low': [traffic_level_low],
            'Traffic_Level_Medium': [traffic_level_medium],
            'Traffic_Level_High': [traffic_level_high],
            'Time_of_Day_Afternoon': [time_of_day_afternoon],
            'Time_of_Day_Evening': [time_of_day_evening],
            'Time_of_Day_Morning': [time_of_day_morning],
            'Time_of_Day_Night': [time_of_day_night],
            'Vehicle_Type_Bike': [vehicle_type_bike],
            'Vehicle_Type_Car': [vehicle_type_car],
            'Vehicle_Type_Scooter': [vehicle_type_scooter],
            'Preparation_Time_min': [preparation_time_min],
            'Courier_Experience_yrs': [courier_experience_yrs]
        }

        input_df = pd.DataFrame(input_data)
        input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)
        prediction = model.predict(input_df)

        return jsonify({"predicted_delivery_time": prediction[0]})

    return render_template("prediction.html", prediction=None)

#3. Statistics Route (Filtered by users)
#Query parameters - weather, traffic, time of day, vehicle type, preparation time, courier experience
#Returns a list of statistics for the given query parameters

@app.route("/statistics", methods=["GET"])
def statistics():
    # Get query parameters
    weather = request.args.get('weather')
    traffic = request.args.get('traffic')
    time_of_day = request.args.get('time_of_day')
    vehicle_type = request.args.get('vehicle_type')
    preparation_time = request.args.get('preparation_time')
    courier_experience = request.args.get('courier_experience')

    # Load the data
    data = pd.read_csv("data/Food_Delivery_times.csv")
    



#Path parameter route
@app.route("/data/<int:order_id>", methods=["GET"])
def data(order_id):
    data = pd.read_csv("data/Food_Delivery_times.csv")
    data['Order_ID'] = data['Order_ID'].astype(int)
    return jsonify(data[data['Order_ID'] == order_id].to_dict(orient="records"))


if __name__ == "__main__":
    app.run(debug=True)
