import numpy as np 
import pandas as pd
import pickle
import math
from flask import Flask, request, jsonify, render_template

app = Flask(__name__, template_folder = "template" ,static_folder="staticfiles") ## assign Flask = app
# Loading the model
cat_final = pickle.load(open(r"capstone_cars24.pkl",'rb'))
# Loading LabelEncoders
le_Car_Name = pickle.load(open(r"le_Car_Name.pkl", 'rb'))
le_RTO = pickle.load(open(r"le_RTO.pkl", 'rb'))

# Loading standardscaler
scale_kmsd = pickle.load(open(r"scale_kmsd.pkl", 'rb'))


uc = pd.read_excel("cleaned_cars24.xlsx")

@app.route('/')
def index():
    car_brands = sorted(list(set([x.split()[0] for x in uc['Car_Name'].unique()])))
    car_names = sorted(uc['Car_Name'].unique())
    Model_Years = sorted(uc.Model_Year.unique())
    states = sorted(list(set([x.split('-')[0] for x in uc['RTO'].unique()])))
    RTOs = sorted(uc['RTO'].unique())
    return render_template('index.html', car_brands=car_brands, car_names=car_names, Model_Years=Model_Years, states=states, RTOs=RTOs)

@app.route('/predict', methods=['POST'])
def predict():
    car_name = request.form.get('Car_Name')
    Car_Name = car_name.strip().lower()
    Model_Year = int(request.form.get('Model_Year'))
    Body_type = int(request.form.get('Body_type'))
    Tranmission = int(request.form.get('Tranmission'))
    Kms_Driven = int(request.form.get('Kms_Driven'))
    rto = request.form.get('RTO')
    RTOn = rto.strip().lower()
    Cruise_control = int(request.form.get('Cruise_control'))
    Steering_mounted_controls = int(request.form.get('Steering_mounted_controls'))
    
    Car_Name = le_Car_Name.transform([Car_Name])[0]
    scaled_Kms_Driven = scale_kmsd.transform(np.array([[Kms_Driven]]))
    RTOn = le_RTO.transform([RTOn])[0]
    
    ip = pd.DataFrame({'Car_Name':Car_Name,'Model_Year':Model_Year,'Body_type':Body_type, 'Tranmission':Tranmission,
                      'Kms_Driven':scaled_Kms_Driven[0][0],'RTO':RTOn, 'Cruise_control':Cruise_control,
                      'Steering_mounted_controls':Steering_mounted_controls}, index = [0])
    print(ip)
    output = cat_final.predict(ip)[0]
    return render_template('index.html',prediction_text="Car price will be around {} lacs".format(math.floor(output)))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
