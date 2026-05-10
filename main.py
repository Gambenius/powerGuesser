import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from src.processor import parse_fit_file
from src.physics import CyclingPhysics
from src.fit_exporter import save_fit_bytes_simple

# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="Pywermeter", page_icon="🚴", layout="wide")
st.title("🚴 Pywermeter.py")

# Sidebar Configuration
st.sidebar.header("Settings")
my_mass = st.sidebar.number_input("Total Mass (kg)", value=73)
my_cda = st.sidebar.slider("CdA (Aero)", 0.20, 0.45, 0.29, step=0.01)
my_crr = st.sidebar.slider("Crr (Rolling)", 0.000, 0.015, 0.005, step=0.0005, format="%.4f")
smoothing_m = st.sidebar.slider("Elevation Smooth (meters)", 5, 50, 20)
speed_smooth_s = st.sidebar.slider("Speed Smooth (seconds)", 1, 10, 5)

uploaded_file = st.file_uploader("Upload your FIT file", type="fit")

def save_to_strava_gpx_string(df):
    ET.register_namespace('', "http://www.topografix.com/GPX/1/1")
    ET.register_namespace('gpxtpx', "http://www.garmin.com/xmlschemas/TrackPointExtension/v1")
    gpx = ET.Element("gpx", {"version": "1.1", "creator": "Pywermeter.py", "xmlns": "http://www.topografix.com/GPX/1/1"})
    trk = ET.SubElement(gpx, "trk")
    trkseg = ET.SubElement(trk, "trkseg")
    for _, row in df.iterrows():
        trkpt = ET.SubElement(trkseg, "trkpt", {"lat": f"{row['lat']:.7f}", "lon": f"{row['lon']:.7f}"})
        ET.SubElement(trkpt, "ele").text = f"{row['ele_smoothed']:.2f}"
        ET.SubElement(trkpt, "time").text = row['time'].strftime('%Y-%m-%dT%H:%M:%SZ')
        ext = ET.SubElement(trkpt, "extensions")
        ET.SubElement(ext, "power").text = str(int(row['p_guessed']))
        tpe = ET.SubElement(ext, "{http://www.garmin.com/xmlschemas/TrackPointExtension/v1}TrackPointExtension")
        if 'hr' in row and not pd.isna(row['hr']):
            ET.SubElement(tpe, "{http://www.garmin.com/xmlschemas/TrackPointExtension/v1}hr").text = str(int(row['hr']))
        if 'cad' in row and not pd.isna(row['cad']):
            ET.SubElement(tpe, "{http://www.garmin.com/xmlschemas/TrackPointExtension/v1}cad").text = str(int(row['cad']))
    return ET.tostring(gpx, encoding='unicode', method='xml')

if uploaded_file is not None:
    with open("temp.fit", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    df = parse_fit_file("temp.fit")
    # --- DEBUG SECTION ---
    with st.expander("🔍 Inspect FIT File Channels (Columns)"):
        st.write(f"**Found {len(df.columns)} channels:**")
        st.write(list(df.columns))
        st.write("**Data Preview (First 5 rows):**")
        st.dataframe(df.head())
    # ---------------------
    # --- DEFINE THIS EARLY AND SAFELY ---
    # We check if 'power' is a column AND if it contains any non-zero/non-null data
    has_real_power = 'real_power' in df.columns and df['real_power'].notnull().any() and df['real_power'].sum() > 0
    # ------------------------------------

    # 1. CORE DATA CALCULATIONS (Fixes the KeyError)
    df['speed_smoothed'] = df['speed'].rolling(window=speed_smooth_s, center=True, min_periods=1).mean()
    df['dist_delta'] = df['speed_smoothed'] * df['dt']
    df['cum_dist_km'] = df['dist_delta'].cumsum() / 1000.0

    # 2. ELEVATION SMOOTHING
    avg_speed = df['speed_smoothed'].mean() if df['speed_smoothed'].mean() > 0 else 5
    rows_in_window = max(int(smoothing_m / avg_speed), 5) 
    df['ele_smoothed'] = df['ele'].rolling(window=rows_in_window, center=True, min_periods=1).mean()

    # 3. PHYSICS ENGINE
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
            powers.append(max(0.0, p))
    
    df['p_guessed'] = powers

    # 4. DASHBOARD METRICS
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate averages safely
    avg_guessed = df['p_guessed'].mean()
    col1.metric("Estimated Avg", f"{avg_guessed:.0f} W")
    
    if has_real_power:
        # Using .mean() on real_power automatically ignores NaNs (missing data)
        avg_real = df['real_power'].mean()
        diff = avg_guessed - avg_real
        
        col2.metric(
            label="Real Avg", 
            value=f"{avg_real:.0f} W", 
            delta=f"{diff:+.1f} W (Error)", # Shows + or - sign
            delta_color="inverse" # Optional: Makes the delta RED if positive (overestimating)
        )
    else:
        col2.metric("Real Avg", "N/A")

    # Keep your distance and elevation metrics
    col3.metric("Distance", f"{df['cum_dist_km'].max():.2f} km")
    col4.metric("Elevation Gain", f"{max(0, ele_diff[ele_diff > 0].sum()):.0f} m")

    # 5. VISUALS
    st.subheader("Power Profile")

    # --- SAMPLING LOGIC ---
    # Define your sample rate (e.g., 5 means taking 1 point every 5 seconds)
    # Automatically adjust sample rate so we never plot more than 2000 points
    total_rows = len(df)
    sample_rate = max(1, total_rows // 400)    
    # Create the dictionary using a sliced DataFrame [::sample_rate]
    chart_dict = {
        'Distance (km)': df['cum_dist_km'][::sample_rate],
        'Estimated Power (W)': df['p_guessed'].rolling(30, center=True).mean()[::sample_rate]
    }

    if has_real_power:
        # Use the same slice to keep the indices aligned
        chart_dict['Original Power (W)'] = df['real_power'].rolling(30, center=True).mean()[::sample_rate]

    plot_df = pd.DataFrame(chart_dict).set_index('Distance (km)')

    # Chart Colors: Estimated (Blue), Original (Red)
    chart_colors = ["#0000FF", "#FF0000"] if has_real_power else ["#0000FF"]
    
    st.line_chart(plot_df, color=chart_colors)

    if has_real_power:
        st.info(f"🔴 **Red**: Original | 🔵 **Blue**: Estimate (Downsampled to 1/{sample_rate} points)")

    # 6. DOWNLOAD SECTION
    st.subheader("📥 Download Results")
    
    col_gpx, col_fit = st.columns(2)
    
    with col_gpx:
        # GPX Download Button
        gpx_str = save_to_strava_gpx_string(df)
        st.download_button(
            label="📩 Download GPX for Strava",
            data=gpx_str,
            file_name=f"{uploaded_file.name.split('.')[0]}_fixed.gpx",
            mime="application/gpx+xml"
        )
    
    with col_fit:
        # FIT Download Button with processed power data
        try:
            fit_bytes = save_fit_bytes_simple("temp.fit", df)
            st.download_button(
                label="💾 Download Processed FIT File",
                data=fit_bytes,
                file_name=f"{uploaded_file.name.split('.')[0]}_processed.fit",
                mime="application/octet-stream"
            )
        except Exception as e:
            st.error(f"⚠️ Error generating FIT file: {str(e)}")
            st.info("You can still download the GPX file above.")
