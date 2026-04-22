import numpy as np
import pandas as pd
from src.processor import parse_fit_file
from src.physics import CyclingPhysics

# --- CONFIGURATION ---
MY_MASS = 85        # kg (Rider + Bike + Gear)
MY_CDA = 0.32       # Aerodynamic drag coefficient
MY_CRR = 0.005      # Rolling resistance coefficient
FIT_FILE = "data/canyon1.fit"

def main():
    print(f"🚀 Loading FIT file: {FIT_FILE}")
    try:
        df = parse_fit_file(FIT_FILE)
    except Exception as e:
        print(f"❌ Failed to parse FIT file: {e}")
        return

    if df.empty:
        print("❌ Error: No data found in FIT file.")
        return

    # Force index cleanup one last time
    df = df.loc[~df.index.duplicated(keep='first')].copy()
    df = df.reset_index(drop=True)

    print(f"📊 Loaded {len(df)} data points.")
    
    # Initialize Physics Engine
    physics = CyclingPhysics(MY_MASS, MY_CDA, MY_CRR)

    print("⚡ Calculating physics vectors...")
    
    # Extract NumPy arrays to bypass Pandas alignment issues
    speed_array = df['speed'].values
    dt_array = df['dt'].values
    ele_array = df['ele'].values
    
    # Use cadence if available, otherwise assume pedaling
    cadence_array = df['cad'].values if 'cad' in df.columns else np.ones(len(df)) * 90

    # 1. Calculate Elevation Delta (Vertical Velocity)
    ele_diff = np.diff(ele_array, prepend=ele_array[0])
    
    # 2. Calculate Distance Delta (Horizontal Velocity)
    dist_diff = speed_array * dt_array
    
    # 3. Calculate Grade (Slope)
    # Using atan2 or simple rise/run. Rise/run is fine for cycling grades.
    grade = np.zeros_like(dist_diff)
    safe_mask = dist_diff > 0.1  # Only calc grade if we moved > 10cm
    grade[safe_mask] = ele_diff[safe_mask] / dist_diff[safe_mask]
    
    # Clip grade to realistic limits (+/- 25%) to filter GPS noise
    df['grade'] = np.clip(grade, -0.25, 0.25)

    print("🚴‍♂️ Simulating power output...")
    powers = [0.0] # First point power is 0

    for i in range(1, len(df)):
        # If cadence is 0, power is 0 (coasting)
        if cadence_array[i] <= 0:
            p = 0.0
        else:
            p = physics.calculate_power(
                v_m_s = speed_array[i],
                v_prev = speed_array[i-1],
                grade = df['grade'].iloc[i],
                dt = dt_array[i]
            )
        powers.append(p)

    df['calculated_power'] = powers

    # --- RESULTS ---
    avg_p = df['calculated_power'].mean()
    max_p = df['calculated_power'].max()
    
    print("-" * 30)
    print(f"✅ CALCULATION COMPLETE")
    print(f"⏱️ Ride Duration: {df['dt'].sum() / 60:.1f} minutes")
    print(f"🚵 Average Power: {avg_p:.2f} W")
    print(f"🚀 Max Power:     {max_p:.2f} W")
    print("-" * 30)

    # Optional: Save a CSV for a quick vibe check in Excel/Sheets
    # df.to_csv("data/processed_ride.csv", index=False)

if __name__ == "__main__":
    main()