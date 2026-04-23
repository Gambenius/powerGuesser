import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from src.processor import parse_fit_file
from src.physics import CyclingPhysics

# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="Pywermeter.py", page_icon="🚴")
st.title("🚴 Pywermeter.py")
st.markdown("Upload a .fit file to calculate physics-based power and export to GPX.")

# Sidebar Configuration
st.sidebar.header("Settings")
my_mass = st.sidebar.number_input("Total Mass (kg)", value=73, help="Rider + Bike + Gear")
my_cda = st.sidebar.slider("CdA (Aero)", 0.20, 0.45, 0.29, step=0.01)
my_crr = st.sidebar.slider("Crr (Rolling)", 0.001, 0.010, 0.005, step=0.001)
smoothing_m = st.sidebar.slider("Elevation Smooth (meters)", 5, 50, 20)
speed_smooth_s = st.sidebar.slider("Speed Smooth (seconds)", 1, 10, 5)

uploaded_file = st.file_uploader("Choose a FIT file", type="fit")

def save_to_gpx_string(df):
    """Generates GPX XML and returns it as a string for downloading."""
    ET.register_namespace('', "http://www.topografix.com/GPX/1/1")
    ET.register_namespace('gpxtpx', "http://www.garmin.com/xmlschemas/TrackPointExtension/v1")
    
    gpx = ET.Element("gpx", {
        "version": "1.1", "creator": "Pywermeter.py",
        "xmlns": "http://www.topografix.com/GPX/1/1",
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance"
    })
    trk = ET.SubElement(gpx, "trk")
    trkseg = ET.SubElement(trk, "trkseg")

    for _, row in df.iterrows():
        trkpt = ET.SubElement(trkseg, "trkpt", {"lat": f"{row['lat']:.7f}", "lon": f"{row['lon']:.7f}"})
        ET.SubElement(trkpt, "ele").text = f"{row['ele_smoothed']:.2f}"
        ET.SubElement(trkpt, "time").text = row['time'].strftime('%Y-%m-%dT%H:%M:%SZ')
        
        extensions = ET.SubElement(trkpt, "extensions")
        ET.SubElement(extensions, "power").text = str(int(row['p_guessed']))
        
        tpe = ET.SubElement(extensions, "{http://www.garmin.com/xmlschemas/TrackPointExtension/v1}TrackPointExtension")
        if 'hr' in row and not pd.isna(row['hr']):
            ET.SubElement(tpe, "{http://www.garmin.com/xmlschemas/TrackPointExtension/v1}hr").text = str(int(row['hr']))
        if 'cad' in row and not pd.isna(row['cad']):
            ET.SubElement(tpe, "{http://www.garmin.com/xmlschemas/TrackPointExtension/v1}cad").text = str(int(row['cad']))

    return ET.tostring(gpx, encoding='unicode', method='xml')

if uploaded_file is not None:
    # Save temp file for processor
    with open("temp.fit", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    df = parse_fit_file("temp.fit")

    # 1. Smoothing
    df['speed_smoothed'] = df['speed'].rolling(window=speed_smooth_s, center=True, min_periods=1).mean()
    avg_speed = df['speed_smoothed'].mean() if df['speed_smoothed'].mean() > 0 else 5
    rows_in_window = max(int(smoothing_m / avg_speed), 5) 
    df['ele_smoothed'] = df['ele'].rolling(window=rows_in_window, center=True, min_periods=1).mean()

    # 2. Physics
    physics = CyclingPhysics(my_mass, my_cda, my_crr)
    v = df['speed_smoothed'].values
    ele_array = df['ele_smoothed'].values
    dt = df['dt'].values
    cadence = df['cad'].values if 'cad' in df.columns else np.ones(len(df)) * 90
    
    ele_diff = np.diff(ele_array, prepend=ele_array[0])
    grade = np.zeros_like(v)
    safe_mask = (v * dt) > 0.1
    grade[safe_mask] = ele_diff[safe_mask] / (v[safe_mask] * dt[safe_mask])
    
    powers = []
    for i in range(len(df)):
        if i == 0 or cadence[i] <= 0:
            powers.append(0.0)
        else:
            p = physics.calculate_power(v[i], v[i-1], grade[i], dt[i])
            powers.append(max(0, p))
    
    df['p_guessed'] = powers

    # 3. Visuals
    st.subheader("Power Profile (30s Moving Average)")
    p_chart = df['p_guessed'].rolling(30, center=True).mean()
    st.line_chart(p_chart)

    # 4. Download
    gpx_str = save_to_strava_gpx_string(df)
    st.download_button(
        label="📩 Download GPX for Strava",
        data=gpx_str,
        file_name=f"{uploaded_file.name.split('.')[0]}_with_power.gpx",
        mime="application/gpx+xml"
    )
    
    st.success(f"Processed {len(df)} points. Average Power: {df['p_guessed'].mean():.1f}W")