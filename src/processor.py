import pandas as pd
from fitparse import FitFile

def parse_fit_file(file_path):
    fitfile = FitFile(file_path)
    data = []

    for record in fitfile.get_messages('record'):
        r_data = {record_data.name: record_data.value for record_data in record}
        data.append(r_data)

    df = pd.DataFrame(data)

    # Priority 1: enhanced_speed (Highest resolution)
    # Priority 2: speed (Wheel sensor usually)
    if 'enhanced_speed' in df.columns and 'speed' in df.columns:
        # If both exist, keep enhanced_speed but rename it
        df = df.drop(columns=['speed']) 
        df = df.rename(columns={'enhanced_speed': 'speed'})
    elif 'enhanced_speed' in df.columns:
        df = df.rename(columns={'enhanced_speed': 'speed'})

    # Standard renames
    df = df.rename(columns={
        'enhanced_altitude': 'ele',
        'timestamp': 'time',
        'position_lat': 'pos_lat',
        'position_long': 'pos_lon',
        'cadence': 'cad'
    })

    # Drop any remaining duplicate columns just in case
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # Keep the essentials
    needed = ['time', 'ele', 'speed', 'pos_lat', 'pos_lon', 'cad']
    df = df[[c for c in needed if c in df.columns]].copy()

    # Drop rows missing crucial physics data (must have speed and elevation)
    df = df.dropna(subset=['speed', 'ele']).reset_index(drop=True)

    # Semicircles to Degrees
    sc_to_deg = 180 / pow(2, 31)
    if 'pos_lat' in df.columns:
        df['lat'] = df['pos_lat'] * sc_to_deg
        df['lon'] = df['pos_lon'] * sc_to_deg

    # Calculate dt
    df['dt'] = df['time'].diff().dt.total_seconds().fillna(1.0)
    df.loc[df['dt'] <= 0, 'dt'] = 1.0

    return df