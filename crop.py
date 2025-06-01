import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('saved_steps.pkl','rb') as file:
        loaded_steps = pickle.load(file)
    return loaded_steps
        
loaded_steps=load_model()

model = loaded_steps['model']
region_encoder = loaded_steps['region_encoder']
soil_encoder = loaded_steps['soil_encoder']
crop_encoder = loaded_steps['crop_encoder']
weather_encoder = loaded_steps['weather_encoder']
scaler = loaded_steps['scaler']

st.title('crop yields prediction')
st.write('lets predict the crop yields')
region=['West', 'South', 'North', 'East']
region_select=st.selectbox('what region you are from?',region)

soiltype=['Sandy', 'Clay', 'Loam', 'Silt', 'Peaty', 'Chalky']
soiltype_select=st.selectbox('what soil are you farming on?',soiltype)

crops=['Cotton', 'Rice', 'Barley', 'Soybean', 'Wheat', 'Maize']
crop_select=st.selectbox('what crop are you farming?',crops)

rainfall=st.slider('how much rainfall in mm do you get in your region?',min_value=90,max_value=1000)
temparature=st.slider('how much temparature in celcious does your region get?',min_value=13,max_value=40)
fertilizer=st.radio('do you use fertilizer during farm?(1 for yes, 0 for no)',options=[1,0])
irrigation=st.radio('do you use irrigation on your crops?(1 for yes, 0 for no)',options=[1,0])

weather=['Cloudy', 'Rainy', 'Sunny']
weather_select=st.selectbox('on average, how was your weather?',weather)

date_to_harvest=st.slider('how many days did you take before harvesting?',min_value=50,max_value=150)



ok=st.button('predict')
if ok:
    try:
        region_encoded = region_encoder.transform([region_select]).reshape(1, -1)
        soil_type_encoded = soil_encoder.transform([soiltype_select]).reshape(1, -1)
        crop_encode = crop_encoder.transform([crop_select]).reshape(1, -1)
        weather_encode = weather_encoder.transform([weather_select]).reshape(1, -1)

        categorical = np.concatenate((region_encoded, soil_type_encoded, crop_encode, weather_encode), axis=1)
        numerical = np.array([[rainfall, temparature, fertilizer, irrigation, date_to_harvest]])

        final_features = np.concatenate((categorical, numerical), axis=1)
        final_features_scaled = scaler.transform(final_features)

        prediction = model.predict(final_features_scaled)

        st.subheader(f'Predicted Crop Yield in tonnes: {prediction[0]:.2f}')
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
