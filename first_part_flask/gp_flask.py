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
    courier_experience = request.args.get('courier_experience')

    # Load the data
    data = pd.read_csv("data/Food_Delivery_times.csv")

    def get_weather_stats(data, weather):
        if weather:
            data = data[data['Weather'] == weather.title()]
        
        weather_stats = {
            "average_distance_km": data["Distance_km"].mean(),
            "average_preparation_time_min": data["Preparation_Time_min"].mean(),
            "average_courier_experience_yrs": data["Courier_Experience_yrs"].mean(),
            "average_delivery_time_min": data["Delivery_Time_min"].mean()
        }
        
        return weather_stats

    def get_traffic_stats(data, traffic):
        if traffic:
            data = data[data['Traffic_Level'] == traffic.title()]
        
        traffic_stats = {
            "average_distance_km": data["Distance_km"].mean(),
            "average_preparation_time_min": data["Preparation_Time_min"].mean(),
            "average_courier_experience_yrs": data["Courier_Experience_yrs"].mean(),
            "average_delivery_time_min": data["Delivery_Time_min"].mean()
        }
        
        return traffic_stats

    def get_time_of_day_stats(data, time_of_day):
        if time_of_day:
            data = data[data['Time_of_Day'] == time_of_day.title()]
        
        time_of_day_stats = {
            "average_distance_km": data["Distance_km"].mean(),
            "average_preparation_time_min": data["Preparation_Time_min"].mean(),
            "average_courier_experience_yrs": data["Courier_Experience_yrs"].mean(),
            "average_delivery_time_min": data["Delivery_Time_min"].mean()
        }
        
        return time_of_day_stats

    def get_vehicle_type_stats(data, vehicle_type):
        if vehicle_type:
            data = data[data['Vehicle_Type'] == vehicle_type.title()]
        
        vehicle_type_stats = {
            "average_distance_km": data["Distance_km"].mean(),
            "average_preparation_time_min": data["Preparation_Time_min"].mean(),
            "average_courier_experience_yrs": data["Courier_Experience_yrs"].mean(),
            "average_delivery_time_min": data["Delivery_Time_min"].mean()
        }
        
        return vehicle_type_stats
    
    def get_courier_experience_stats(data, courier_experience):
        data["Courier_Experience_Group"] = pd.cut(data["Courier_Experience_yrs"], bins=[0, 1, 3, 5, 10], labels=["0-1", "1-3", "3-5", "5-10"])
        
        if courier_experience is not None:
            if 0 < courier_experience <= 1:
                experience_group = "0-1"
            elif 1 < courier_experience <= 3:
                experience_group = "1-3"
            elif 3 < courier_experience <= 5:
                experience_group = "3-5"
            elif 5 < courier_experience <= 10:
                experience_group = "5-10"
            else:
                return None

            data = data[data["Courier_Experience_Group"] == experience_group]
            
            courier_experience_stats = {
                "average_distance_km": data["Distance_km"].mean(),
                "average_preparation_time_min": data["Preparation_Time_min"].mean(),
                "average_courier_experience_yrs": data["Courier_Experience_yrs"].mean(),
                "average_delivery_time_min": data["Delivery_Time_min"].mean()
            }
            
            return courier_experience_stats
    
    stats = {}

    if weather:
        w_stats = get_weather_stats(data, weather)
        stats[f"weather_stats_{weather}"] = w_stats
    if traffic:
        t_stats = get_traffic_stats(data, traffic)
        stats[f"traffic_stats_{traffic}"] = t_stats
    if time_of_day:
        tod_stats = get_time_of_day_stats(data, time_of_day)
        stats[f"time_of_day_stats_{time_of_day}"] = tod_stats
    if vehicle_type:
        v_stats = get_vehicle_type_stats(data, vehicle_type)
        stats[f"vehicle_type_stats_{vehicle_type}"] = v_stats
    if courier_experience:
        c_stats = get_courier_experience_stats(data, courier_experience)
        stats[f"courier_experience_stats_{courier_experience}"] = c_stats

    return jsonify(stats)
    
#Path parameter route
@app.route("/data/<int:order_id>", methods=["GET"])
def data(order_id):
    data = pd.read_csv("data/Food_Delivery_times.csv")
    data['Order_ID'] = data['Order_ID'].astype(int)
    return jsonify(data[data['Order_ID'] == order_id].to_dict(orient="records"))


if __name__ == "__main__":
    app.run(debug=True)
