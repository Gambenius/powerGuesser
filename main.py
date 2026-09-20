import streamlit as st
import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import tempfile
from src.processor import parse_fit_file, write_power_fit, prepare_activity
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
optimize_clicked = st.sidebar.button("Optimize CdA from real power")
st.sidebar.caption('Calibration fixes Crr at 0.003 and keeps your mass fixed.')


@st.cache_data(max_entries=4, show_spinner=False)
def read_upload(contents):
    with tempfile.NamedTemporaryFile(suffix='.fit') as source:
        source.write(contents)
        source.flush()
        return parse_fit_file(source.name)


@st.cache_data(max_entries=4, show_spinner=False)
def process_upload(contents, elevation_window, speed_window):
    return prepare_activity(read_upload(contents), elevation_window, speed_window)


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
    start = None
    last_up = 0

    def finish():
        if start is None:
            return
        total_distance = distance[last_up] - distance[start]
        grade = (elevation[last_up] - elevation[start]) / max(total_distance, 1)
        if total_distance >= minimum_distance_m and grade > minimum_grade:
            weights = df['analysis_dt'].to_numpy()[start + 1:last_up + 1]
            average_power = np.average(power[start + 1:last_up + 1], weights=weights)
            climbs.append({
                'start_index': start,
                'end_index': last_up,
                'distance_km': total_distance / 1000,
                'grade': grade,
                'average_power': round(average_power),
            })
    for index in range(1, len(df)):
        if not df['valid_interval'].iloc[index]:
            finish()
            start = None
            continue
        rising = elevation[index] > elevation[index - 1] + 1e-8
        gap_end = index - 1 if rising else index
        if start is not None and distance[gap_end] - distance[last_up] > 100 + 1e-8:
            finish()
            start = None
        if rising:
            if start is None:
                start = index - 1
            last_up = index
    finish()
    return climbs

uploaded_file = st.file_uploader("Upload your FIT file", type="fit")

if uploaded_file is not None:
    contents = uploaded_file.getvalue()
    try:
        df = process_upload(contents, smoothing_m, speed_smooth_s)
    except ValueError as error:
        st.error(str(error))
        st.stop()
    if not df['valid_interval'].any():
        st.error('No usable speed/elevation intervals of 5 seconds or less.')
        st.stop()
    
    # --- DEFINE THIS EARLY AND SAFELY ---
    # We check if 'power' is a column AND if it contains any non-zero/non-null data
    has_real_power = 'real_power' in df.columns and df['real_power'].notnull().any() and df['real_power'].sum() > 0
    # ------------------------------------

    # 1. CORE DATA CALCULATIONS (Fixes the KeyError)

    # 2. ELEVATION SMOOTHING

    # 3. PHYSICS ENGINE
    physics = CyclingPhysics(my_mass, my_cda, my_crr)
    v = df['speed_smoothed'].values
    ele_array = df['ele_smoothed'].values
    dt = df['analysis_dt'].to_numpy()
    cadence = df['cad'].fillna(90).to_numpy()
    
    ele_diff = df['elevation_delta'].to_numpy()
    distance_delta = df['dist_delta'].to_numpy()
    
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
    valid_power = np.isfinite(powers) & (dt > 0)
    if has_real_power:
        valid_power &= df['real_power'].notna().to_numpy()
    weights = dt[valid_power]
    low_delta = low_power[valid_power] - powers[valid_power]
    high_delta = high_power[valid_power] - powers[valid_power]

    uncertainty_w = None
    if has_real_power:
        measured_power = df['real_power'].to_numpy()
        measured_filtered = lowpass_power(measured_power, dt)
        estimated_filtered = lowpass_power(df['p_guessed'].to_numpy(), dt)
        residual = estimated_filtered - measured_filtered
        valid_residual = residual[np.isfinite(residual) & valid_power]
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
        st.info('No climbs of at least 1 km above 3% average grade were found (maximum flat/downhill gap: 100 m).')

    if optimize_clicked:
        if not has_real_power:
            st.sidebar.warning("Upload a FIT file containing real power first.")
        else:
            measured_power = df['real_power'].to_numpy()
            if np.count_nonzero(np.isfinite(measured_power) & (dt > 0)) < 30:
                st.warning('Calibration needs at least 30 valid measured-power intervals.')
                st.stop()
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
                "Crr (fixed)": f"{optimized[2]:.3f}",
            })
            st.sidebar.caption("± ranges correspond to approximately ±5 W average modeled power.")
            st.sidebar.caption('Use the suggested CdA together with Crr = 0.003 in the sidebar. Calibrate using solo rides with varied speeds.')

    # 4. DASHBOARD METRICS
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Calculate averages safely
    avg_guessed = float(np.average(powers[valid_power], weights=weights)) if valid_power.any() else 0
    if valid_power.any():
        average_low_delta = float(np.average(low_delta, weights=weights))
        average_high_delta = float(np.average(high_delta, weights=weights))
        uncertainty_label = f"{average_low_delta:+.0f} / {average_high_delta:+.0f} W"
        col1.metric("Estimated Avg", f"{avg_guessed:.0f} W", delta=uncertainty_label, delta_color="off")
    else:
        col1.metric("Estimated Avg", f"{avg_guessed:.0f} W")
    
    if has_real_power:
        # Using .mean() on real_power automatically ignores NaNs (missing data)
        avg_real = float(np.average(df['real_power'].to_numpy()[valid_power], weights=weights)) if valid_power.any() else 0
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
        with st.expander("Measured-power validation"):
            uncertainty_col1, uncertainty_col2, uncertainty_col3 = st.columns(3)
            uncertainty_col1.metric("MAE", f"{np.mean(np.abs(valid_residual)):.0f} W")
            uncertainty_col2.metric("Bias", f"{np.mean(valid_residual):+.0f} W")
            uncertainty_col3.metric("75th-percentile error", f"±{uncertainty_w:.0f} W")
            st.caption("This is the 75th percentile of the absolute error between low-pass filtered estimated and measured power.")

    with st.expander("Road-bike parameter uncertainty"):
        if valid_power.any():
            low_average = float(np.average(low_delta, weights=weights))
            high_average = float(np.average(high_delta, weights=weights))
            low_p75 = -float(np.percentile(np.abs(low_delta), 75))
            high_p75 = float(np.percentile(np.abs(high_delta), 75))
            st.write(f"Relative ranges: CdA ±20% ({cda_range[0]:.3f}–{cda_range[1]:.3f} m²), Crr ±10% ({crr_range[0]:.4f}–{crr_range[1]:.4f}).")
            range_col1, range_col2 = st.columns(2)
            range_col1.metric("Average power range", f"{low_average:+.0f} / {high_average:+.0f} W")
            range_col2.metric("75th-percentile range", f"{low_p75:+.0f} / {high_p75:+.0f} W")
            st.caption('Parameter sensitivity only: excludes wind, drafting, elevation error and braking. Average bounds use the same time weights as Estimated Avg; percentiles describe individual samples.')
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

    with st.expander('Data quality and processing'):
        st.write(df.attrs.get('quality', {}))
        st.caption('Smoothing is confined to continuous valid data. Elevation uses distance; speed uses time. Averages include zeros and exclude gaps over 5 s. Measured-power comparisons use matching samples. Distance excludes unobserved gaps.')
        st.line_chart(df[['cum_dist_km', 'ele', 'ele_smoothed']].iloc[::sample_rate].set_index('cum_dist_km'))

    with st.expander("🔍 Inspect FIT File Channels (Columns)"):
        st.write(f"**Found {len(df.columns)} channels:**")
        st.write(list(df.columns))
        st.write("**Data Preview (First 5 rows):**")
        st.dataframe(df.head())
