import streamlit as st
import pandas as pd
import joblib
from datetime import time

st.set_page_config(page_title="Kanakapura Traffic AI", page_icon="🚦", layout="wide")

@st.cache_resource
def load_backend():
    model = joblib.load('traffic_model.pkl')
    columns = joblib.load('model_columns.pkl')
    
    df = pd.read_csv('Master_Traffic_Data.csv')
    hv_col = 'HVs_Encountered' if 'HVs_Encountered' in df.columns else 'HVs Encountered'
    lv_col = 'LVs_Encountered' if 'LVs_Encountered' in df.columns else 'LVs Encountered'
    avg_total_vehicles = df[hv_col].mean() + df[lv_col].mean()
    
    return model, columns, avg_total_vehicles

try:
    model, training_columns, avg_total_vehicles = load_backend()
except FileNotFoundError:
    st.error("⚠️ Backend files missing! Please run train_model.py first.")
    st.stop()

st.title("🚦 Kanakapura Road Traffic Forecaster")
st.markdown("##### Multi-Milestone AI Prediction Model")
st.divider()

col_input, col_result = st.columns([1.2, 1])

with col_input:
    st.subheader("📋 Input Travel Conditions")
    with st.container(border=True):
        vehicle_class = st.selectbox("Select Vehicle Type", ['Car', 'Truck', 'Van', 'Bus', 'Bike', 'Auto', 'Scooter'])
        
        # --- THE UPGRADE: User-Friendly Clock Slider ---
        departure_time = st.slider(
            "Departure Time (Evening Peak)", 
            min_value=time(17, 0), # 5:00 PM
            max_value=time(19, 0), # 7:00 PM
            value=time(17, 30),    # Default 5:30 PM
            format="hh:mm a"
        )
        # Convert clock time to elapsed seconds for the math model
        start_time_sec = (departure_time.hour - 17) * 3600 + departure_time.minute * 60
        # -----------------------------------------------
            
        bottleneck_delay = st.slider("Active Bottleneck Delay (Seconds)", 0, 60, 5)
        
    st.info(f"📊 **Data Insight:** Model utilizing Kanakapura dataset average of **{int(avg_total_vehicles)} total vehicles** for density.")

with col_result:
    st.subheader("⏱️ Predicted ETAs")
    st.markdown("<br>", unsafe_allow_html=True) 
    
    if st.button("Forecast Commute Timeline", type="primary", use_container_width=True):
        
        input_dict = {
            'Start Time (Elapsed Sec)': start_time_sec,
            'Start_Time_Elapsed_Sec': start_time_sec, 
            'Bottleneck Delay (sec)': bottleneck_delay,
            'Bottleneck_Delay_Sec': bottleneck_delay, 
            'Total Vehicles': avg_total_vehicles
        }
        
        for col in training_columns:
            if col.startswith('Class_'):
                class_name = col.replace('Class_', '')
                input_dict[col] = 1 if class_name.lower() == vehicle_class.lower() else 0
                
        final_input_dict = {k: v for k, v in input_dict.items() if k in training_columns}
        input_df = pd.DataFrame([final_input_dict], columns=training_columns)
        input_df = input_df.fillna(0)
        
        # --- THE UPGRADE: Extracting Both Predictions ---
        predictions = model.predict(input_df)[0]
        pred_500m = predictions[0]
        pred_1km = predictions[1]
        
        # Physics Floor
        if pred_1km < pred_500m + bottleneck_delay + 15:
            pred_1km = pred_500m + bottleneck_delay + 15
        
        # Display Dual Metrics side-by-side
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            with st.container(border=True):
                st.metric("ETA to 500m", f"{int(pred_500m)} Sec", delta=f"{pred_500m/60:.1f} Min", delta_color="off")
        with metric_col2:
            with st.container(border=True):
                st.metric("ETA to 1km", f"{int(pred_1km)} Sec", delta=f"{pred_1km/60:.1f} Min", delta_color="off")
                
        st.success("✅ Multi-output prediction generated successfully.")
    else:
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            with st.container(border=True):
                st.metric("ETA to 500m", "-- Sec", delta="-- Min", delta_color="off")
        with metric_col2:
            with st.container(border=True):
                st.metric("ETA to 1km", "-- Sec", delta="-- Min", delta_color="off")
        st.caption("Select your departure time and click forecast.")