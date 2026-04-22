import numpy as np
import pandas as pd
from src.processor import parse_fit_file
from src.physics import CyclingPhysics

# --- CONFIGURATION ---
MY_MASS = 85        # kg (Rider + Bike + Gear)
MY_CDA = 0.25       # Aerodynamic drag coefficient
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

    # Cleanup index
    df = df.loc[~df.index.duplicated(keep='first')].copy()
    df = df.reset_index(drop=True)

    print(f"📊 Loaded {len(df)} data points.")
    
    # --- ELEVATION SMOOTHING ---
    # GPS elevation is noisy. We smooth it over a 5-second window to fix the "spikes"
    df['ele_smoothed'] = df['ele'].rolling(window=5, min_periods=1, center=True).mean()

    # Initialize Physics Engine
    physics = CyclingPhysics(MY_MASS, MY_CDA, MY_CRR)

    print("⚡ Calculating physics vectors...")
    
    speed_array = df['speed'].values
    dt_array = df['dt'].values
    ele_array = df['ele_smoothed'].values # Using smoothed elevation
    cadence_array = df['cad'].values if 'cad' in df.columns else np.ones(len(df)) * 90

    # 1. Calculate Elevation and Grade
    ele_diff = np.diff(ele_array, prepend=ele_array[0])
    dist_diff = speed_array * dt_array
    
    grade = np.zeros_like(dist_diff)
    safe_mask = dist_diff > 0.1
    grade[safe_mask] = ele_diff[safe_mask] / dist_diff[safe_mask]
    df['grade'] = np.clip(grade, -0.25, 0.25)

    print("🚴‍♂️ Simulating power output...")
    powers = [0.0]

    for i in range(1, len(df)):
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

    # --- POWER PERCENTILES (POWER CURVE) ---
    percentiles = [25, 50, 75, 90, 95, 99]
    
    print("-" * 45)
    print(f"{'Metric':<15} | {'Real (W)':<10} | {'Guessed (W)':<12}")
    print("-" * 45)
    
    avg_calc = df['calculated_power'].mean()
    if 'real_power' in df.columns:
        avg_real = df['real_power'].mean()
        print(f"{'Average':<15} | {avg_real:<10.2f} | {avg_calc:<12.2f}")
        for p in percentiles:
            val_real = np.percentile(df['real_power'], p)
            val_calc = np.percentile(df['calculated_power'], p)
            print(f"{str(p) + 'th %':<15} | {val_real:<10.2f} | {val_calc:<12.2f}")
    else:
        print(f"{'Average':<15} | {'N/A':<10} | {avg_calc:<12.2f}")
        for p in percentiles:
            val_calc = np.percentile(df['calculated_power'], p)
            print(f"{str(p) + 'th %':<15} | {'N/A':<10} | {val_calc:<12.2f}")
            
    print("-" * 45)
    
    if 'real_power' in df.columns:
        diff = ((avg_calc - avg_real) / avg_real) * 100 if avg_real > 0 else 0
        print(f"⚖️ Overall Bias: {diff:+.1f}%")
        if diff > 10:
            print("💡 Suggestion: Lower your MY_CDA or check your total weight.")
        elif diff < -10:
            print("💡 Suggestion: Increase your MY_CDA (too much aero?)")

if __name__ == "__main__":
    main()