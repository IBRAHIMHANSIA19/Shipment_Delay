import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import predict
import requests

# Set page config for a premium look
st.set_page_config(
    page_title="Shipment Delay Prediction Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS styling for premium look (dark theme accents, rounded corners)
st.markdown("""
<style>
    /* Import modern Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [data-testid="stAppViewContainer"], .main {
        font-family: 'Inter', -apple-system, sans-serif !important;
    }
    
    .main-header {
        font-size: 2.2rem;
        color: #0F172A;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin-bottom: 0.25rem;
    }
    
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    
    /* Modern white cards for expanders with drop shadows */
    div[data-testid="stExpander"] {
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important;
        background-color: #FFFFFF !important;
        margin-bottom: 1.25rem !important;
        overflow: hidden;
    }
    
    .streamlit-expanderHeader {
        background-color: #FFFFFF !important;
        font-weight: 600 !important;
        color: #0F172A !important;
        padding: 0.75rem 1rem !important;
        font-size: 1.05rem !important;
        border-bottom: 1px solid #F1F5F9 !important;
    }
    
    /* Input field styling adjustments */
    div[data-baseweb="input"], div[data-baseweb="select"] {
        border-radius: 8px !important;
    }
    
    /* Premium button styles with dynamic gradients & hover effects */
    div[data-testid="stButton"] button {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.8rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.25) !important;
        transition: all 0.2s ease-in-out !important;
    }
    
    div[data-testid="stButton"] button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.35) !important;
        background: linear-gradient(135deg, #1D4ED8 0%, #2563EB 100%) !important;
    }
    
    div[data-testid="stButton"] button:active {
        transform: translateY(1px) !important;
    }
    
    /* Clean metrics look */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #0F172A !important;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        color: #64748B !important;
        font-weight: 500 !important;
    }
    
    /* Prediction outcome cards styling */
    .status-delayed {
        background-color: #FEF2F2;
        color: #991B1B;
        padding: 1rem;
        border-radius: 10px;
        font-weight: 700;
        font-size: 1.25rem;
        text-align: center;
        border: 1px solid #FEE2E2;
        border-left: 6px solid #EF4444;
        box-shadow: 0 2px 4px rgba(239, 68, 68, 0.05);
    }
    
    .status-ontime {
        background-color: #ECFDF5;
        color: #065F46;
        padding: 1rem;
        border-radius: 10px;
        font-weight: 700;
        font-size: 1.25rem;
        text-align: center;
        border: 1px solid #D1FAE5;
        border-left: 6px solid #10B981;
        box-shadow: 0 2px 4px rgba(16, 185, 129, 0.05);
    }
    
    /* Section headers */
    h3 {
        color: #1D4ED8 !important;
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.75rem !important;
        letter-spacing: -0.02em !important;
    }
    
    /* Sidebar styling & container customization */
    [data-testid="stSidebar"] {
        background-color: #F8FAFC !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #1E3A8A !important;
        font-weight: 700 !important;
    }
    
    /* Sidebar List/Bullet Styling */
    .sidebar-bullet-list {
        list-style-type: none !important;
        padding-left: 0 !important;
        margin-top: 0.5rem !important;
        margin-bottom: 1rem !important;
    }
    .sidebar-bullet-list li {
        position: relative !important;
        padding-left: 1.5rem !important;
        margin-bottom: 0.75rem !important;
        color: #475569 !important;
        font-size: 0.88rem !important;
        line-height: 1.5 !important;
    }
    .sidebar-bullet-list li::before {
        content: "•" !important;
        color: #3B82F6 !important; /* Premium Blue bullet */
        font-weight: bold !important;
        font-size: 1.4rem !important;
        position: absolute !important;
        left: 0.2rem !important;
        top: -0.15rem !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    """Loads prediction model and columns lists, cached for fast response."""
    model = predict.load_xgboost_model("best_xgboost_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, feature_columns

CITIES = {
    "Mumbai (Maharashtra)": "Mumbai",
    "New Delhi (Delhi)": "New Delhi",
    "Bengaluru (Karnataka)": "Bengaluru",
    "Chennai (Tamil Nadu)": "Chennai",
    "Kolkata (West Bengal)": "Kolkata",
    "Hyderabad (Telangana)": "Hyderabad",
    "Ahmedabad (Gujarat)": "Ahmedabad",
    "Pune (Maharashtra)": "Pune",
    "Surat (Gujarat)": "Surat",
    "Jaipur (Rajasthan)": "Jaipur",
    "Lucknow (Uttar Pradesh)": "Lucknow",
    "Kanpur (Uttar Pradesh)": "Kanpur",
    "Nagpur (Maharashtra)": "Nagpur",
    "Indore (Madhya Pradesh)": "Indore",
    "Bhopal (Madhya Pradesh)": "Bhopal",
    "Visakhapatnam (Andhra Pradesh)": "Visakhapatnam",
    "Patna (Bihar)": "Patna",
    "Vadodara (Gujarat)": "Vadodara",
    "Ludhiana (Punjab)": "Ludhiana",
    "Agra (Uttar Pradesh)": "Agra",
    "Nashik (Maharashtra)": "Nashik",
    "Varanasi (Uttar Pradesh)": "Varanasi",
    "Srinagar (Jammu & Kashmir)": "Srinagar",
    "Amritsar (Punjab)": "Amritsar",
    "Ranchi (Jharkhand)": "Ranchi",
    "Coimbatore (Tamil Nadu)": "Coimbatore",
    "Vijayawada (Andhra Pradesh)": "Vijayawada",
    "Jodhpur (Rajasthan)": "Jodhpur",
    "Madurai (Tamil Nadu)": "Madurai",
    "Raipur (Chhattisgarh)": "Raipur",
    "Guwahati (Assam)": "Guwahati",
    "Chandigarh (Punjab/Haryana)": "Chandigarh",
    "Kochi (Kerala)": "Kochi",
    "Dehradun (Uttarakhand)": "Dehradun",
    "Bhubaneswar (Odisha)": "Bhubaneswar"
}

def fetch_google_weather(city_name):
    try:
        import urllib.parse
        import re
        query = urllib.parse.quote_plus(f"weather in {city_name}")
        url = f"https://www.google.com/search?q={query}&hl=en"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
        }
        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code == 200:
            html = res.text
            
            # Extract attributes from Google's weather card snippet
            temp_match = re.search(r'id="wob_tm"[^>]*>(\d+)<', html) or re.search(r'class="wob_t"[^>]*>(\d+)[°<]', html)
            desc_match = re.search(r'id="wob_dc"[^>]*>([^<]+)<', html)
            humidity_match = re.search(r'id="wob_hm"[^>]*>(\d+)%<', html)
            wind_match = re.search(r'id="wob_ws"[^>]*>([^<]+)<', html)
            precip_match = re.search(r'id="wob_pp"[^>]*>(\d+)%<', html)
            
            if temp_match:
                temp = float(temp_match.group(1))
                cond_desc = desc_match.group(1) if desc_match else "Clear"
                humidity = float(humidity_match.group(1)) if humidity_match else 50.0
                
                # Parse wind speed
                wind_str = wind_match.group(1) if wind_match else "10 km/h"
                wind_nums = re.findall(r'\d+', wind_str)
                wind_speed = float(wind_nums[0]) if wind_nums else 10.0
                
                # Parse precipitation probability and map to rainfall estimate
                precip = float(precip_match.group(1)) if precip_match else 0.0
                rainfall = (precip / 100.0) * 5.0
                
                desc = cond_desc.lower()
                if any(x in desc for x in ['rain', 'shower', 'drizzle', 'rainy', 'thunderstorm', 'storm']):
                    cond = "Rain" if 'storm' not in desc else "Storm"
                elif any(x in desc for x in ['snow', 'ice', 'sleet', 'blizzard', 'snowy']):
                    cond = "Snow"
                elif any(x in desc for x in ['storm', 'thunder', 'lightning', 'cyclone', 'tornado', 'squall']):
                    cond = "Storm"
                elif any(x in desc for x in ['fog', 'mist', 'haze', 'dust', 'sand', 'smoke', 'overcast']):
                    if 'overcast' in desc or 'cloudy' in desc:
                        cond = "Cloudy"
                    else:
                        cond = "Fog"
                elif any(x in desc for x in ['cloud', 'cloudy', 'partly cloudy', 'mostly cloudy']):
                    cond = "Cloudy"
                else:
                    cond = "Clear"
                
                # Estimate visibility
                if cond == "Clear":
                    visibility = 10.0
                elif cond == "Cloudy":
                    visibility = 9.0
                elif cond == "Rain":
                    visibility = 6.0
                elif cond == "Fog":
                    visibility = 1.5
                else:
                    visibility = 4.0
                    
                return {
                    "success": True,
                    "temperature": temp,
                    "visibility": visibility,
                    "humidity": humidity,
                    "wind_speed": wind_speed,
                    "rainfall": rainfall,
                    "weather_condition": cond,
                    "raw_desc": cond_desc,
                    "source": "Google Weather"
                }
    except Exception as e:
        pass
    return {"success": False}

def fetch_wttr_weather(city_name):
    try:
        url = f"https://wttr.in/{city_name.replace(' ', '+')}?format=j1"
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            curr = data['current_condition'][0]
            temp = float(curr['temp_C'])
            visibility = float(curr['visibility'])
            humidity = float(curr['humidity'])
            wind_speed = float(curr['windspeedKmph'])
            rainfall = float(curr.get('precipMM', 0.0))
            
            desc = curr['weatherDesc'][0]['value'].lower()
            
            if any(x in desc for x in ['rain', 'shower', 'drizzle', 'rainy']):
                cond = "Rain"
            elif any(x in desc for x in ['snow', 'ice', 'sleet', 'blizzard', 'snowy']):
                cond = "Snow"
            elif any(x in desc for x in ['storm', 'thunder', 'lightning', 'cyclone', 'tornado', 'squall']):
                cond = "Storm"
            elif any(x in desc for x in ['fog', 'mist', 'haze', 'dust', 'sand', 'smoke']):
                cond = "Fog"
            elif any(x in desc for x in ['cloud', 'overcast', 'cloudy', 'misty']):
                cond = "Cloudy"
            else:
                cond = "Clear"
                
            return {
                "success": True,
                "temperature": temp,
                "visibility": visibility,
                "humidity": humidity,
                "wind_speed": wind_speed,
                "rainfall": rainfall,
                "weather_condition": cond,
                "raw_desc": curr['weatherDesc'][0]['value'],
                "source": "wttr.in Fallback"
            }
    except Exception as e:
        pass
    return {"success": False}

def fetch_live_weather(city_name):
    # 1. Try Google Weather first
    res = fetch_google_weather(city_name)
    if res["success"]:
        return res
        
    # 2. Fall back to wttr.in
    return fetch_wttr_weather(city_name)

try:
    model, feature_columns = load_assets()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)

# Sidebar layout
st.sidebar.markdown('<div style="font-size: 1.8rem; font-weight: 800; color: #1E3A8A; margin-top: 1rem; margin-bottom: 1rem;">📦 Navigation</div>', unsafe_allow_html=True)
app_mode = st.sidebar.radio("Select Application Mode", ["🔍 Single Shipment Forecast", "📁 Batch CSV Processing"])

# Sidebar Quick Guide with bullets
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 System Insights")
st.sidebar.markdown("""
<ul class="sidebar-bullet-list">
    <li><strong>Model Engine:</strong> XGBoost Classifier</li>
    <li><strong>Live Weather:</strong> Auto-scraping from live hubs</li>
    <li><strong>Geolocations:</strong> Major Indian and global hubs</li>
    <li><strong>Batch Processing:</strong> Seamless CSV batch forecast</li>
    <li><strong>Data Standard:</strong> Built-in schema verification</li>
</ul>
""", unsafe_allow_html=True)

# Set default threshold
threshold = 0.5

if not model_loaded:
    st.error(f"⚠️ **Error Loading Model Assets!**")
    st.write(f"Ensure that `best_xgboost_model.pkl` and `feature_columns.pkl` are located in your workspace directory.")
    st.info(f"Error details: `{model_error}`")
    st.stop()

# Header without logo images
st.markdown('<div class="main-header">🚢 Shipment Delay Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered logistics risk optimization and delay probability forecast.</div>', unsafe_allow_html=True)

# ----------------- Single Prediction mode -----------------
if app_mode == "🔍 Single Shipment Forecast":
    st.markdown("### 📝 Enter Shipment Attributes")
    st.write("Grouped below are key indicators affecting transit time. Review and adjust details as needed.")

    # We display all features stacked sequentially in a single vertical column with a compact 2-column-per-line structure
    st.markdown("### 📦 Cargo & Priority")
    with st.expander("📦 Cargo Details", expanded=True):
        cr1_col1, cr1_col2 = st.columns(2)
        with cr1_col1:
            cargo_options = ["-- Select Cargo Type --"] + list(predict.LABEL_MAPPINGS['cargo_type'].keys())
            cargo_type = st.selectbox("Cargo Type", cargo_options, index=0)
        with cr1_col2:
            weight_kg = st.number_input("Weight (kg)", min_value=0.0, value=0.0, step=100.0)
            
        cr2_col1, cr2_col2 = st.columns(2)
        with cr2_col1:
            volume_cbm = st.number_input("Volume (cbm)", min_value=0.0, value=0.0, step=1.0)
        with cr2_col2:
            declared_value = st.number_input("Declared Value ($)", min_value=0.0, value=0.0, step=500.0)
            
        cr3_col1, cr3_col2 = st.columns(2)
        with cr3_col1:
            weight_per_unit = st.number_input("Weight per unit (kg)", min_value=0.0, value=0.0, step=0.5)
        with cr3_col2:
            priority_options = ["-- Select Priority --"] + [opt for opt in predict.LABEL_MAPPINGS['priority'].keys() if opt != 'Urgent']
            priority = st.selectbox("Priority", priority_options, index=0)
            
        st.write("")
        paperwork_completed = st.checkbox("Paper Work Completed", value=False)
        
        # Map paperwork completeness to underlying features
        documentation_complete = paperwork_completed
        insurance = paperwork_completed
        customs_required = not paperwork_completed
        fragile_x = False
        fragile_y = False
        hazardous = False
        perishable = False
        temperature_controlled = False
        inspection_required = False

    st.markdown("### 🛣️ Route Details")
    with st.expander("🛣️ Route Details", expanded=True):
        rr1_col1, rr1_col2 = st.columns(2)
        with rr1_col1:
            mode_options = ["-- Select Shipping Mode --"] + list(predict.LABEL_MAPPINGS['shipping_mode'].keys())
            shipping_mode = st.selectbox("Shipping Mode", mode_options, index=0)
        with rr1_col2:
            distance_km = st.number_input("Transit Distance (km)", min_value=0.0, value=0.0, step=50.0)
            
        rr2_col1, rr2_col2 = st.columns(2)
        with rr2_col1:
            average_transit_days = st.number_input("Estimated Transit Days", min_value=0, value=0, step=1)
        with rr2_col2:
            risk_options = ["-- Select Route Risk Level --"] + list(predict.LABEL_MAPPINGS['route_risk'].keys())
            route_risk = st.selectbox("Route Risk Level", risk_options, index=0)
            
        traffic_index = st.slider("Traffic Congestion Index", 0.0, 100.0, 0.0, 1.0)

    st.markdown("### 🌦️ Live Weather")
    selected_city = st.selectbox("Select Shipping Hub City", ["-- Select Shipping Hub City --"] + list(CITIES.keys()), index=0)
    
    # Fetch the live weather for the selected city
    if selected_city != "-- Select Shipping Hub City --":
        weather_info = fetch_live_weather(CITIES[selected_city])
    else:
        weather_info = {"success": False}
    
    if weather_info["success"]:
        st.metric(label="Weather State", value=f"{weather_info['weather_condition']} ({weather_info['raw_desc']})")
        
        w_col1, w_col2 = st.columns(2)
        with w_col1:
            st.metric(label="Temp", value=f"{weather_info['temperature']}°C")
            st.metric(label="Wind", value=f"{weather_info['wind_speed']} km/h")
        with w_col2:
            st.metric(label="Visibility", value=f"{weather_info['visibility']} km")
            st.metric(label="Humidity", value=f"{weather_info['humidity']}%")
            
        weather_condition = weather_info["weather_condition"]
        temperature = weather_info["temperature"]
        visibility = weather_info["visibility"]
        humidity = weather_info["humidity"]
        wind_speed = weather_info["wind_speed"]
        rainfall = weather_info["rainfall"]
    else:
        if selected_city != "-- Select Shipping Hub City --":
            st.warning("⚠️ Could not fetch live weather data. Using default weather values.")
        weather_condition = "Clear"
        temperature = 0.0
        visibility = 0.0
        humidity = 0.0
        wind_speed = 0.0
        rainfall = 0.0

    # Show Carrier & Fleet Attributes permanently with 2-column-per-line structure
    st.write("")
    st.markdown("### 🚚 Carrier & Fleet Attributes")
    with st.expander("🚚 Fleet Specifications", expanded=True):
        fr1_col1, fr1_col2 = st.columns(2)
        with fr1_col1:
            average_rating = st.slider("Carrier Average Rating", 0.0, 5.0, 0.0, 0.1)
        with fr1_col2:
            years_of_service = st.number_input("Carrier Years of Service", min_value=0, value=0, step=1)
            
        fr2_col1, fr2_col2 = st.columns(2)
        with fr2_col1:
            # Map Shipping Mode to default Vehicle Type option
            mode_to_vehicle = {
                "Air": "Aircraft",
                "Rail": "Wagon",
                "Road": "Truck",
                "Sea": "Ship"
            }
            default_vehicle = mode_to_vehicle.get(shipping_mode, None)
            vehicle_options = ["-- Select Vehicle Type --"] + list(predict.LABEL_MAPPINGS['vehicle_type'].keys())
            
            if default_vehicle and default_vehicle in vehicle_options:
                default_vehicle_index = vehicle_options.index(default_vehicle)
            else:
                default_vehicle_index = 0
                
            vehicle_type = st.selectbox("Vehicle Type", vehicle_options, index=default_vehicle_index)
        with fr2_col2:
            maintenance_options = ["-- Select Maintenance Status --"] + list(predict.LABEL_MAPPINGS['maintenance_status'].keys())
            maintenance_status = st.selectbox("Vehicle Maintenance Status", maintenance_options, index=0)
            
        # Auto-map capacity based on Vehicle Type
        vehicle_capacity_mapping = {
            "Aircraft": 50000.0,
            "Ship": 150000.0,
            "Truck": 20000.0,
            "Wagon": 60000.0
        }
        capacity_kg = vehicle_capacity_mapping.get(vehicle_type, 0.0)
        
        vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, value=0, step=1)

    # Initialize input_dict with all 51 features
    input_dict = {
        'shipping_mode': shipping_mode,
        'shipment_type': predict.FEATURE_DEFAULTS.get('shipment_type', 1), # Default fallback
        'priority': priority,
        'weight_kg': weight_kg,
        'volume_cbm': volume_cbm,
        'declared_value': declared_value,
        'insurance': insurance,
        'fragile_x': fragile_x,
        'carrier_type': shipping_mode,
        'average_rating': average_rating,
        'fleet_size': predict.FEATURE_DEFAULTS.get('fleet_size', 100),
        'years_of_service': years_of_service,
        'customer_type': predict.FEATURE_DEFAULTS.get('customer_type', 0), # Default fallback
        'industry': predict.FEATURE_DEFAULTS.get('industry', 4), # Default fallback
        'country': predict.FEATURE_DEFAULTS.get('country', 20), # Default fallback
        'customer_status': predict.FEATURE_DEFAULTS.get('customer_status', 0), # Default fallback
        'customs_required': customs_required,
        'documentation_complete': documentation_complete,
        'inspection_required': inspection_required,
        'cargo_type': cargo_type,
        'category': predict.FEATURE_DEFAULTS.get('category', 3), # Default fallback
        'hs_code': predict.FEATURE_DEFAULTS.get('hs_code', 8517.0),
        'hazardous': hazardous,
        'perishable': perishable,
        'temperature_controlled': temperature_controlled,
        'fragile_y': fragile_y,
        'weight_per_unit': weight_per_unit,
        'distance_km': distance_km,
        'average_transit_days': average_transit_days,
        'route_risk': route_risk,
        'traffic_index': traffic_index,
        'vehicle_type': vehicle_type,
        'capacity_kg': capacity_kg,
        'fuel_type': predict.FEATURE_DEFAULTS.get('fuel_type', 1), # Default fallback
        'maintenance_status': maintenance_status,
        'vehicle_age': vehicle_age,
        'warehouse_capacity': predict.FEATURE_DEFAULTS.get('warehouse_capacity', 50000), # Default fallback
        'current_utilization': predict.FEATURE_DEFAULTS.get('current_utilization', 75.0), # Default fallback
        'warehouse_type': predict.FEATURE_DEFAULTS.get('warehouse_type', 1), # Default fallback
        'weather_condition': weather_condition,
        'temperature': temperature,
        'rainfall': rainfall,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'visibility': visibility,
        'booking_month': predict.FEATURE_DEFAULTS.get('booking_month', 6), # Default fallback
        'booking_day': predict.FEATURE_DEFAULTS.get('booking_day', 15), # Default fallback
        'booking_weekday': predict.FEATURE_DEFAULTS.get('booking_weekday', 2), # Default fallback
        'ship_month': predict.FEATURE_DEFAULTS.get('ship_month', 6), # Default fallback
        'ship_day': predict.FEATURE_DEFAULTS.get('ship_day', 16), # Default fallback
        'ship_weekday': predict.FEATURE_DEFAULTS.get('ship_weekday', 3) # Default fallback
    }

    st.markdown("---")
    
    # Prediction trigger
    if st.button("🔍 Forecast Shipment Risk Status", type="primary", use_container_width=True):
        # 1. Validation logic for placeholders
        unselected_fields = []
        if cargo_type == "-- Select Cargo Type --":
            unselected_fields.append("Cargo Type")
        if priority == "-- Select Priority --":
            unselected_fields.append("Priority")
        if shipping_mode == "-- Select Shipping Mode --":
            unselected_fields.append("Shipping Mode")
        if route_risk == "-- Select Route Risk Level --":
            unselected_fields.append("Route Risk Level")
        if selected_city == "-- Select Shipping Hub City --":
            unselected_fields.append("Shipping Hub City")
        if vehicle_type == "-- Select Vehicle Type --":
            unselected_fields.append("Vehicle Type")
        if maintenance_status == "-- Select Maintenance Status --":
            unselected_fields.append("Vehicle Maintenance Status")
            
        # 2. Validation logic for zero numeric fields
        zero_fields = []
        if weight_kg <= 0.0:
            zero_fields.append("Weight (kg)")
        if volume_cbm <= 0.0:
            zero_fields.append("Volume (cbm)")
        if declared_value <= 0.0:
            zero_fields.append("Declared Value ($)")
        if distance_km <= 0.0:
            zero_fields.append("Transit Distance (km)")
        if average_transit_days <= 0:
            zero_fields.append("Estimated Transit Days")
            
        if unselected_fields or zero_fields:
            st.error("⚠️ **Validation Error! Please fill in and select all required fields.**")
            if unselected_fields:
                st.markdown("**The following selections are missing:**")
                st.markdown("\n".join([f"- Please select a valid **{field}**." for field in unselected_fields]))
            if zero_fields:
                st.markdown("**The following values must be greater than zero:**")
                st.markdown("\n".join([f"- Please enter a non-zero value for **{field}**." for field in zero_fields]))
        else:
            res = predict.predict_delay(input_dict, model, feature_columns, predict.LABEL_MAPPINGS, threshold)
        
        if res["Status"] == "Success":
            st.markdown("### 📊 Prediction Result")
            
            p_label = res["Prediction"]
            conf = res["Confidence (%)"]
            
            r_col1, r_col2 = st.columns(2)
            
            with r_col1:
                if p_label == "Delayed":
                    st.markdown(f'<div class="status-delayed">🛑 DELAYED RISK IDENTIFIED</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="status-ontime">✅ ON-TIME ARRIVAL EXPECTED</div>', unsafe_allow_html=True)
                    
                st.write("")
                st.metric(label="Prediction Confidence Score", value=f"{conf:.2f}%")
                
            with r_col2:
                # Show breakdown chart/table
                breakdown = pd.DataFrame({
                    "Outcome": ["On-Time", "Delayed"],
                    "Probability (%)": [res["On-Time Probability (%)"], res["Delayed Probability (%)"]]
                })
                st.dataframe(breakdown.set_index("Outcome"), use_container_width=True)
                st.progress(res["Delayed Probability (%)"] / 100.0)
                st.caption(f"Risk Probability: {res['Delayed Probability (%)']:.2f}% (Threshold: {threshold * 100:.1f}%)")
        else:
            st.error(f"Prediction Failed: {res['Message']}")

# ----------------- Batch CSV Prediction mode -----------------
elif app_mode == "📁 Batch CSV Processing":
    st.markdown("### 📥 Batch Processing (CSV)")
    st.write("Upload a CSV file containing multiple shipment records to predict delays in bulk.")
    
    # Download template helper
    st.markdown("#### 1. Download CSV Schema Template")
    st.write("The CSV file must contain columns corresponding to the shipment features (e.g. shipping_mode, weight_kg, etc.). Missing columns will automatically be filled with default values.")
    
    # Create template DataFrame with defaults
    template_df = pd.DataFrame([predict.FEATURE_DEFAULTS])
    template_csv = template_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Sample CSV Template",
        data=template_csv,
        file_name="shipment_template.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    st.markdown("#### 2. Upload and Process Your CSV")
    uploaded_file = st.file_uploader("Upload Shipment CSV File", type=["csv"])
    
    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(input_df)} records.")
            st.dataframe(input_df.head(10), use_container_width=True)
            
            if st.button("🚀 Execute Batch Predictions", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                predictions = []
                on_time_probs = []
                delayed_probs = []
                confidences = []
                
                # Perform predictions row-by-row
                for idx, row in input_df.iterrows():
                    row_dict = row.to_dict()
                    res = predict.predict_delay(row_dict, model, feature_columns, predict.LABEL_MAPPINGS, threshold)
                    if res["Status"] == "Success":
                        predictions.append(res["Prediction"])
                        on_time_probs.append(res["On-Time Probability (%)"])
                        delayed_probs.append(res["Delayed Probability (%)"])
                        confidences.append(res["Confidence (%)"])
                    else:
                        predictions.append("ERROR")
                        on_time_probs.append(0.0)
                        delayed_probs.append(0.0)
                        confidences.append(0.0)
                    progress_bar.progress((idx + 1) / len(input_df))
                    
                # Append predicted fields to input dataframe
                output_df = input_df.copy()
                output_df["predicted_status"] = predictions
                output_df["on_time_probability_pct"] = on_time_probs
                output_df["delayed_probability_pct"] = delayed_probs
                output_df["prediction_confidence_pct"] = confidences
                
                st.markdown("#### 📊 Prediction Summaries")
                
                sum_col1, sum_col2 = st.columns(2)
                
                with sum_col1:
                    delayed_count = sum(1 for p in predictions if p == "Delayed")
                    total_count = len(predictions)
                    delayed_rate = (delayed_count / total_count) * 100 if total_count > 0 else 0
                    
                    st.metric("Total Shipments Analyzed", f"{total_count}")
                    st.metric("Flagged Delayed Risk", f"{delayed_count} ({delayed_rate:.1f}%)")
                    
                with sum_col2:
                    status_counts = pd.Series(predictions).value_counts()
                    st.write("Outcome Distribution")
                    st.bar_chart(status_counts)
                    
                st.markdown("#### 📥 Download Results Table")
                st.dataframe(output_df.head(20), use_container_width=True)
                
                output_csv = output_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Completed Predictions CSV",
                    data=output_csv,
                    file_name="shipment_predictions_output.csv",
                    mime="text/csv",
                    type="primary"
                )
        except Exception as e:
            st.error(f"Error reading or processing CSV: {e}")
