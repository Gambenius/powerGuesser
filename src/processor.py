import pandas as pd
from fitparse import FitFile

def parse_fit_file(file_path):
    # 1. Load the binary FIT file
    fitfile = FitFile(file_path)
    data = []

    # 2. Iterate through 'record' messages
    for record in fitfile.get_messages('record'):
        r_data = {record_data.name: record_data.value for record_data in record}
        data.append(r_data)

    df = pd.DataFrame(data)

    # 3. Rename with a priority system
    # We map 'power' (real sensor) to 'real_power' and 'cadence' to 'cad'
    rename_map = {
        'enhanced_altitude': 'ele',
        'enhanced_speed': 'speed',
        'timestamp': 'time',
        'position_lat': 'pos_lat',
        'position_long': 'pos_lon',
        'cadence': 'cad',
        'power': 'real_power'
    }
    
    # Rename columns carefully
    for old_name, new_name in rename_map.items():
        if old_name in df.columns:
            # Avoid duplicate speed columns if both speed and enhanced_speed exist
            if new_name in df.columns and new_name == 'speed':
                continue 
            df = df.rename(columns={old_name: new_name})

    # 4. Force unique columns to resolve the broadcast/duplicate label errors
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # 5. Select only the columns we need
    needed = ['time', 'ele', 'speed', 'pos_lat', 'pos_lon', 'cad', 'real_power']
    df = df[[c for c in needed if c in df.columns]].copy()

    # 6. Clean and Reset
    df = df.dropna(subset=['speed', 'ele']).reset_index(drop=True)
    
    if 'real_power' in df.columns:
        df['real_power'] = df['real_power'].fillna(0)

    # 7. Semicircles to Degrees
    sc_to_deg = 180 / pow(2, 31)
    if 'pos_lat' in df.columns:
        df['lat'] = df['pos_lat'] * sc_to_deg
        df['lon'] = df['pos_lon'] * sc_to_deg

    # 8. Calculate time delta
    df['dt'] = df['time'].diff().dt.total_seconds().fillna(1.0)
    df.loc[df['dt'] <= 0, 'dt'] = 1.0

    return df