import streamlit as st
import pandas as pd
import joblib

try:
    model = joblib.load('traffic_model.pkl')
    training_columns = joblib.load('model_columns.pkl')
except FileNotFoundError:
    st.error("Model files not found! Please run train_model.py first.")
    st.stop()

st.title("🚦 Kanakapura Road Traffic Predictor")
st.markdown("Predict final 1km commute times based on real-time street friction.")
st.divider()

st.sidebar.header("Input Travel Conditions")

vehicle_class = st.sidebar.selectbox("Vehicle Type", ['Car', 'Truck', 'Van', 'Bus', 'Bike', 'Auto', 'Scooter'])
start_time_sec = st.sidebar.slider("Start Time (Seconds elapsed into peak hour)", 0, 3600, 1800, step=60)
time_to_500m = st.sidebar.number_input("Time to clear first 500m (seconds)", min_value=30, max_value=400, value=120)
bottleneck_delay = st.sidebar.slider("Delay at Bottleneck (seconds)", 0, 60, 5)

# Keep the two sliders for UX, but add them together for the model
hvs = st.sidebar.slider("Heavy Vehicles Encountered", 10, 150, 60)
lvs = st.sidebar.slider("Light Vehicles Encountered", 10, 150, 70)
total_vehicles = hvs + lvs

if st.button("Predict 1km Commute Time", type="primary"):
    input_dict = {
        'Start Time (Elapsed Sec)': start_time_sec,
        'Time to 500m (sec)': time_to_500m,
        'Bottleneck Delay (sec)': bottleneck_delay,
        'Total Vehicles': total_vehicles
    }
    
    for col in training_columns:
        if col.startswith('Class_'):
            class_name = col.replace('Class_', '')
            input_dict[col] = 1 if class_name.lower() == vehicle_class.lower() else 0
            
    input_df = pd.DataFrame([input_dict], columns=training_columns)
    input_df = input_df.fillna(0)
    
    pred_seconds = model.predict(input_df)[0]
    
    # Physics Floor
    absolute_minimum = time_to_500m + bottleneck_delay + 15
    if pred_seconds < absolute_minimum:
        pred_seconds = absolute_minimum
    
    st.success("Prediction generated using Multiple Linear Regression.")
    col1, col2 = st.columns(2)
    col1.metric("Predicted Time (Seconds)", f"{int(pred_seconds)} sec")
    col2.metric("Predicted Time (Minutes)", f"{pred_seconds/60:.2f} min")