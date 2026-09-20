import streamlit as st
import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import tempfile
from src.processor import parse_fit_file, write_power_fit
from src.physics import CyclingPhysics, lowpass_power, optimize_parameters


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
ftp = st.sidebar.slider("FTP (W)", 50, 500, 220, step=1)
optimize_clicked = st.sidebar.button("Optimize from real power")


@st.cache_data(max_entries=4, show_spinner=False)
def read_upload(contents):
    with tempfile.NamedTemporaryFile(suffix='.fit') as source:
        source.write(contents)
        source.flush()
        return parse_fit_file(source.name)


@st.cache_data(max_entries=4, show_spinner=False)
def export_upload(contents, powers, indices):
    with tempfile.NamedTemporaryFile(suffix='.fit') as source, tempfile.NamedTemporaryFile(suffix='.fit') as output:
        source.write(contents)
        source.flush()
        write_power_fit(source.name, output.name, powers, indices)
        return output.read()


@st.cache_data(max_entries=4, show_spinner=False)
def find_long_climbs(df, minimum_grade=0.03, minimum_distance_m=1000, max_power_cv=0.35):
    climbs = []
    distance = df['cum_dist_km'].to_numpy() * 1000
    elevation = df['ele_smoothed'].to_numpy()
    power = df['p_guessed'].to_numpy()
    index = 0

    while index < len(df) - 1:
        candidate_end = index + 1
        non_climb_distance = 0.0
        while candidate_end < len(df):
            step_distance = max(0.0, distance[candidate_end] - distance[candidate_end - 1])
            elevation_change = elevation[candidate_end] - elevation[candidate_end - 1]
            if elevation_change < -0.5:
                non_climb_distance += step_distance
            else:
                non_climb_distance = 0.0
            if non_climb_distance > 100:
                candidate_end = max(index + 1, candidate_end - 1)
                break
            candidate_end += 1
        else:
            candidate_end = len(df) - 1

        total_distance = distance[candidate_end] - distance[index]
        grade = (elevation[candidate_end] - elevation[index]) / max(total_distance, 1)
        if total_distance < minimum_distance_m or grade < minimum_grade:
            index += 1
            continue

        segment_power = power[index:candidate_end + 1]
        average_power = float(np.mean(segment_power))
        power_cv = float(np.std(segment_power) / average_power) if average_power > 0 else np.inf
        if power_cv <= max_power_cv:
            climbs.append({
                'start_index': index,
                'end_index': candidate_end,
                'distance_km': total_distance / 1000,
                'grade': grade,
                'average_power': round(average_power),
            })
            index = candidate_end + 1
        else:
            index += 1
    return climbs

uploaded_file = st.file_uploader("Upload your FIT file", type="fit")

if uploaded_file is not None:
    contents = uploaded_file.getvalue()
    df = read_upload(contents)
    
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
    distance_delta = np.maximum(0, (v + np.roll(v, 1)) / 2 * dt)
    distance_delta[0] = 0
    
    powers = physics.estimate_series(v, ele_diff, distance_delta, dt, cadence)

    df['p_guessed'] = powers
    cda_range = (my_cda * 0.80, my_cda * 1.20)
    crr_range = (my_crr * 0.90, my_crr * 1.10)
    low_power = CyclingPhysics(my_mass, cda_range[0], crr_range[0]).estimate_series(
        v, ele_diff, distance_delta, dt, cadence
    )
    high_power = CyclingPhysics(my_mass, cda_range[1], crr_range[1]).estimate_series(
        v, ele_diff, distance_delta, dt, cadence
    )
    valid_power = np.isfinite(powers) & (powers > 0)
    low_delta = low_power[valid_power] - powers[valid_power]
    high_delta = high_power[valid_power] - powers[valid_power]

    uncertainty_w = None
    if has_real_power:
        measured_power = df['real_power'].interpolate(limit_direction='both').to_numpy()
        measured_filtered = lowpass_power(measured_power, dt)
        estimated_filtered = lowpass_power(df['p_guessed'].to_numpy(), dt)
        residual = estimated_filtered - measured_filtered
        valid_residual = residual[np.isfinite(residual) & (measured_filtered > 0)]
        if len(valid_residual):
            uncertainty_w = float(np.percentile(np.abs(valid_residual), 75))

    st.subheader("Recorded Route and Power")
    zone_limits = np.array([0, 0.55, 0.75, 0.87, 0.94, 1.05, 1.20, np.inf]) * ftp
    map_indices = np.unique(np.linspace(0, len(df) - 1, min(len(df), 10000), dtype=int))
    route = df.iloc[map_indices][['lat', 'lon', 'p_guessed']].dropna().copy()
    ftp_zone = np.digitize(route['p_guessed'].to_numpy(), zone_limits[1:-1], right=False) + 1
    route['zone'] = [f'Z{zone}' for zone in ftp_zone]
    climbs = find_long_climbs(df)
    zone_colors = ['#2E86DE', '#2ECC71', '#F1C40F', '#F39C12', '#E74C3C', '#8E44AD', '#922B21']
    zone_domain = [f'Z{zone}' for zone in range(1, 8)]
    layers = []
    view = pdk.ViewState(latitude=float(df['lat'].mean()), longitude=float(df['lon'].mean()), zoom=11)
    coordinates = route[['lon', 'lat']].to_numpy().tolist()
    paths = []
    start = 0
    for end in range(1, len(route)):
        if end == len(route) - 1 or ftp_zone[end] != ftp_zone[start]:
            color = zone_colors[ftp_zone[start] - 1]
            paths.append({'path': coordinates[start:end + 1], 'color': [int(color[offset:offset + 2], 16) for offset in (1, 3, 5)]})
            start = end
    layers.append(pdk.Layer(
        'PathLayer', data=paths, get_path='path', get_color='color',
        get_width=4, width_units='"pixels"', width_min_pixels=4, width_max_pixels=4,
        cap_rounded=True, joint_rounded=True,
    ))
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, map_style='https://basemaps.cartocdn.com/gl/positron-gl-style/style.json'))
    if climbs:
        st.subheader("Climbs")
        climb_table = pd.DataFrame(climbs).assign(
            **{
                'Start distance (km)': lambda table: table['start_index'].map(df['cum_dist_km']),
                'End distance (km)': lambda table: table['end_index'].map(df['cum_dist_km']),
                'Length (km)': lambda table: table['distance_km'],
                'Grade (%)': lambda table: table['grade'] * 100,
                'AVG Power (W)': lambda table: table['average_power'],
            }
        )[
            ['Start distance (km)', 'End distance (km)', 'Length (km)', 'Grade (%)', 'AVG Power (W)']
        ]
        st.dataframe(
            climb_table.style.format({
                'Start distance (km)': '{:.2f}',
                'End distance (km)': '{:.2f}',
                'Length (km)': '{:.2f}',
                'Grade (%)': '{:.1f}',
                'AVG Power (W)': '{:.0f}',
            }),
            hide_index=True,
        )
    else:
        st.info("No climbs of at least 1 km at 3% average grade with steady power were found.")

    if optimize_clicked:
        if not has_real_power:
            st.sidebar.warning("Upload a FIT file containing real power first.")
        else:
            measured_power = df['real_power'].interpolate(limit_direction='both').to_numpy()
            optimized, uncertainty, _, _ = optimize_parameters(
                [my_mass, my_cda, my_crr],
                measured_power,
                v,
                ele_diff,
                distance_delta,
                dt,
                cadence,
            )
            st.sidebar.success("Optimization complete")
            st.sidebar.write({
                "Mass (kg, fixed)": round(optimized[0], 2),
                "CdA": f"{optimized[1]:.4f} ± {uncertainty[0]:.4f}",
                "Crr": f"{optimized[2]:.5f} ± {uncertainty[1]:.5f}",
            })
            st.sidebar.caption("± ranges correspond to approximately ±5 W average modeled power.")

    # 4. DASHBOARD METRICS
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Calculate averages safely
    avg_guessed = df['p_guessed'].mean()
    if valid_power.any():
        average_low_delta = float(np.mean(low_delta))
        average_high_delta = float(np.mean(high_delta))
        uncertainty_label = f"{average_low_delta:+.0f} / {average_high_delta:+.0f} W"
        col1.metric("Estimated Avg", f"{avg_guessed:.0f} W", delta=uncertainty_label, delta_color="off")
    else:
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
    if 'hr' in df.columns and df['hr'].notna().any():
        col5.metric("Avg HR", f"{df['hr'].mean():.0f} bpm")
    else:
        col5.metric("Avg HR", "N/A")

    if uncertainty_w is not None:
        with st.expander("Power uncertainty analysis"):
            uncertainty_col1, uncertainty_col2, uncertainty_col3 = st.columns(3)
            uncertainty_col1.metric("MAE", f"{np.mean(np.abs(valid_residual)):.0f} W")
            uncertainty_col2.metric("Bias", f"{np.mean(valid_residual):+.0f} W")
            uncertainty_col3.metric("75th-percentile error", f"±{uncertainty_w:.0f} W")
            st.caption("This is the 75th percentile of the absolute error between low-pass filtered estimated and measured power.")

    with st.expander("Road-bike parameter uncertainty"):
        if valid_power.any():
            low_average = float(np.mean(low_delta))
            high_average = float(np.mean(high_delta))
            low_p75 = -float(np.percentile(np.abs(low_delta), 75))
            high_p75 = float(np.percentile(np.abs(high_delta), 75))
            st.write(f"Relative ranges: CdA ±20% ({cda_range[0]:.3f}–{cda_range[1]:.3f} m²), Crr ±10% ({crr_range[0]:.4f}–{crr_range[1]:.4f}).")
            range_col1, range_col2 = st.columns(2)
            range_col1.metric("Average power range", f"{low_average:+.0f} / {high_average:+.0f} W")
            range_col2.metric("75th-percentile range", f"{low_p75:+.0f} / {high_p75:+.0f} W")
            st.caption("The first value is the low-end change and the second is the high-end change for 75% of samples.")
        else:
            st.info("No moving samples are available for uncertainty analysis.")

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
    chart_columns = ['Estimated Power (W)']
    if has_real_power:
        chart_columns.append('Original Power (W)')
    power_chart = alt.Chart(plot_df.reset_index()).transform_fold(
        chart_columns, as_=['Series', 'Power (W)']
    ).mark_line().encode(
        x=alt.X('Distance (km):Q', scale=alt.Scale(domain=[0, float(df['cum_dist_km'].max())], domainMin=0, domainMax=float(df['cum_dist_km'].max())), title='Distance (km)'),
        y=alt.Y('Power (W):Q', title='Power (W)'),
        color=alt.Color('Series:N', scale=alt.Scale(range=['#0000FF', '#FF0000']), title=None, legend=alt.Legend(orient='bottom')),
        tooltip=['Distance (km):Q', 'Series:N', 'Power (W):Q'],
    ).interactive()
    st.altair_chart(power_chart, use_container_width=True)

    if has_real_power:
        st.info(f"🔴 **Red**: Original | 🔵 **Blue**: Estimate (Downsampled to 1/{sample_rate} points)")

    if 'hr' in df.columns and df['hr'].notna().any():
        st.subheader("Heart Rate")
        heart_rate_chart = pd.DataFrame({
            'Distance (km)': df['cum_dist_km'][::sample_rate],
            'Heart rate (bpm)': df['hr'].rolling(15, center=True, min_periods=1).mean()[::sample_rate],
        })
        heart_rate_plot = alt.Chart(heart_rate_chart).mark_line(color='#C0392B').encode(
            x=alt.X('Distance (km):Q', scale=alt.Scale(domain=[0, float(df['cum_dist_km'].max())], domainMin=0, domainMax=float(df['cum_dist_km'].max())), title='Distance (km)'),
            y=alt.Y('Heart rate (bpm):Q', title='Heart rate (bpm)'),
            tooltip=['Distance (km):Q', 'Heart rate (bpm):Q'],
        ).interactive()
        st.altair_chart(heart_rate_plot, use_container_width=True)

    # 6. DOWNLOAD
    fit_bytes = export_upload(contents, df['p_guessed'].to_numpy(), df['fit_record_index'].to_numpy())

    st.download_button(
        label="📩 Download FIT for Strava",
        data=fit_bytes,
        file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_power.fit",
        mime="application/octet-stream"
    )

    st.subheader("Time in Power Zones")
    zone_numbers = np.digitize(df['p_guessed'].to_numpy(), zone_limits[1:-1], right=False) + 1
    active_seconds = np.where(dt <= 5, dt, 0)
    zone_seconds = np.bincount(zone_numbers - 1, weights=active_seconds, minlength=7)
    zone_labels = [f"Z{zone}" for zone in range(1, 8)]
    zone_colors = ['#2E86DE', '#2ECC71', '#F1C40F', '#F39C12', '#E74C3C', '#8E44AD', '#922B21']

    zone_chart = pd.DataFrame({'Zone': zone_labels, 'Time (minutes)': np.array(zone_seconds) / 60})
    chart = alt.Chart(zone_chart).mark_bar().encode(
        x=alt.X('Zone:N', sort=zone_labels, title='Power zone'),
        y=alt.Y('Time (minutes):Q', title='Time (minutes)'),
        color=alt.Color('Zone:N', scale=alt.Scale(domain=zone_labels, range=zone_colors), legend=None),
        tooltip=['Zone:N', alt.Tooltip('Time (minutes):Q', format='.1f')],
    ).properties(title=f'Power zones based on FTP = {ftp} W').interactive()
    st.altair_chart(chart, use_container_width=True)

    with st.expander("🔍 Inspect FIT File Channels (Columns)"):
        st.write(f"**Found {len(df.columns)} channels:**")
        st.write(list(df.columns))
        st.write("**Data Preview (First 5 rows):**")
        st.dataframe(df.head())
