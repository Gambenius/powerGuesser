import pandas as pd
from fitparse import FitFile

def parse_fit_file(file_path):
    print(f"📖 Parsing FIT file for all sensors...")
    fitfile = FitFile(file_path)
    data = []
    for record in fitfile.get_messages('record'):
        r_data = {record_data.name: record_data.value for record_data in record}
        data.append(r_data)

    df = pd.DataFrame(data)
    
    # Updated map to include HR and Temp
    rename_map = {
        'enhanced_altitude': 'ele',
        'enhanced_speed': 'speed',
        'timestamp': 'time',
        'position_lat': 'pos_lat',
        'position_long': 'pos_lon',
        'cadence': 'cad',
        'heart_rate': 'hr',
        'temperature': 'temp',
        'power': 'real_power'
    }
    
    for old_name, new_name in rename_map.items():
        if old_name in df.columns:
            # Avoid conflict if 'speed' already exists
            if new_name in df.columns and old_name != 'enhanced_speed': continue 
            df = df.rename(columns={old_name: new_name})

    df = df.loc[:, ~df.columns.duplicated()].copy()
    
    # Keep the new sensor columns
    needed = ['time', 'ele', 'speed', 'pos_lat', 'pos_lon', 'cad', 'hr', 'temp', 'real_power']
    df = df[[c for c in needed if c in df.columns]].copy()
    
    df = df.dropna(subset=['pos_lat']).reset_index(drop=True)

    sc_to_deg = 180 / pow(2, 31)
    df['lat'] = df['pos_lat'] * sc_to_deg
    df['lon'] = df['pos_lon'] * sc_to_deg
    
    df['dt'] = df['time'].diff().dt.total_seconds().fillna(1.0)
    df.loc[df['dt'] <= 0, 'dt'] = 1.0
    
    return df