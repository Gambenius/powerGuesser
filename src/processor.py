import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from fitparse import FitFile
from fit_tool.fit_file import FitFile as WritableFitFile
from fit_tool.fit_file_builder import FitFileBuilder
from fit_tool.definition_message import DefinitionMessage
from fit_tool.data_message import DataMessage
from fit_tool.field_definition import FieldDefinition
from fit_tool.profile.messages.record_message import RecordMessage, RecordPowerField

def parse_fit_file(file_path):
    print(f"📖 Parsing FIT file for all sensors...")
    fitfile = FitFile(file_path)
    data = []
    for record in fitfile.get_messages('record'):
        r_data = {record_data.name: record_data.value for record_data in record}
        data.append(r_data)

    df = pd.DataFrame(data)
    if df.empty or 'timestamp' not in df:
        raise ValueError('The FIT file has no timestamped activity records.')
    df['fit_record_index'] = range(len(df))
    for enhanced, standard in [('enhanced_speed', 'speed'), ('enhanced_altitude', 'altitude')]:
        if enhanced in df:
            df[standard] = df[enhanced].combine_first(df.get(standard, pd.Series(np.nan, index=df.index)))
            df = df.drop(columns=enhanced)
    
    # Updated map to include HR and Temp
    rename_map = {
        'altitude': 'ele',
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
    needed = ['fit_record_index', 'time', 'ele', 'speed', 'distance', 'pos_lat', 'pos_lon', 'cad', 'hr', 'temp', 'real_power']
    df = df[[c for c in needed if c in df.columns]].copy()
    
    for column in ['ele', 'speed', 'pos_lat', 'pos_lon', 'cad']:
        if column not in df:
            df[column] = np.nan

    sc_to_deg = 180 / pow(2, 31)
    df['lat'] = df['pos_lat'] * sc_to_deg
    df['lon'] = df['pos_lon'] * sc_to_deg
    
    df['dt'] = df['time'].diff().dt.total_seconds()
    
    return df


def prepare_activity(df, elevation_window=20, speed_window=5):
    """Process contiguous valid intervals without bridging sensor gaps or pauses."""
    df = df.copy()
    valid = df['dt'].between(0, 5, inclusive='right')
    good = df['speed'].between(0, 40) & df['ele'].notna() & df['time'].notna()
    valid &= good & good.shift(fill_value=False)
    df['segment'] = (~valid).cumsum()
    df['speed_smoothed'] = df['speed']
    df['ele_smoothed'] = df['ele']
    df['dist_delta'] = 0.0
    df['elevation_delta'] = 0.0
    spikes = 0
    for _, group in df.groupby('segment', sort=False):
        if len(group) < 2 or not good.loc[group.index].all():
            continue
        seconds = (group['time'] - group['time'].iloc[0]).dt.total_seconds().to_numpy()
        speed = group['speed'].to_numpy()
        speed_grid = np.arange(0, seconds[-1] + 1, 1.0)
        speed_uniform = np.interp(speed_grid, seconds, speed)
        window = min(int(speed_window) // 2 * 2 + 1, len(speed_grid) // 2 * 2 - 1)
        if window >= 3:
            speed_uniform = savgol_filter(speed_uniform, window, min(2, window - 1))
        smooth_speed = np.maximum(0, np.interp(seconds, speed_grid, speed_uniform))
        delta = np.r_[0, (speed[1:] + speed[:-1]) / 2 * np.diff(seconds)]
        if 'distance' in group:
            recorded = group['distance'].diff().to_numpy()
            usable = np.isfinite(recorded) & (recorded >= 0) & (recorded <= 40 * group['dt'].to_numpy())
            delta = np.where(usable, recorded, delta)
        distance = np.cumsum(delta)
        elevation = group['ele'].to_numpy().copy()
        if len(elevation) >= 3:
            middle = (elevation[:-2] + elevation[2:]) / 2
            isolated = (np.abs(elevation[1:-1] - middle) > 5) & (np.abs(elevation[2:] - elevation[:-2]) < 2)
            spikes += int(isolated.sum())
            elevation[1:-1] = np.where(isolated, middle, elevation[1:-1])
        unique = pd.DataFrame({'distance': distance, 'elevation': elevation}).groupby('distance')['elevation'].median()
        if distance[-1] > 0 and len(unique) >= 3:
            grid = np.linspace(0, distance[-1], max(3, int(distance[-1] / 2) + 1))
            profile = np.interp(grid, unique.index, unique.values)
            window = min(max(3, int(elevation_window / (grid[1] - grid[0])) // 2 * 2 + 1), len(grid) // 2 * 2 - 1)
            if window >= 3:
                profile = savgol_filter(profile, window, 1)
            elevation = np.interp(distance, grid, profile)
        df.loc[group.index, 'speed_smoothed'] = smooth_speed
        df.loc[group.index, 'ele_smoothed'] = elevation
        df.loc[group.index, 'dist_delta'] = delta
        df.loc[group.index, 'elevation_delta'] = np.r_[0, np.diff(elevation)]
    df['valid_interval'] = valid
    df['analysis_dt'] = df['dt'].where(valid, 0.0)
    df['cum_dist_km'] = df['dist_delta'].cumsum() / 1000
    df.attrs['quality'] = {
        'Excluded intervals': int((~valid).sum()),
        'Gaps over 5 seconds': int((df['dt'] > 5).sum()),
        'Missing or invalid speed/elevation': int((~good).sum()),
        'Missing cadence (assumed pedaling)': int(df['cad'].isna().sum()),
        'Isolated elevation spikes repaired': spikes,
    }
    return df


def write_power_fit(file_path, output_path, powers, record_indices):
    """Copy a FIT activity and add calculated power to every record."""
    power_by_record = {
        int(record_index): int(round(max(0, power)))
        for record_index, power in zip(record_indices, powers)
    }
    fit_file = WritableFitFile.from_file(file_path)
    builder = FitFileBuilder(auto_define=True)
    record_index = 0

    def ensure_power_field(definition):
        if definition and not definition.get_field_definition(RecordPowerField.ID):
            definition.add_field_definition(FieldDefinition.from_field(RecordPowerField(size=2)))
            definition.size = DefinitionMessage.calculate_size(
                definition.field_definitions, definition.developer_field_definitions
            )

    for fit_record in fit_file.records:
        message = fit_record.message
        if isinstance(message, DefinitionMessage):
            continue
        if isinstance(message, DataMessage) and message.global_id == RecordMessage.ID:
            ensure_power_field(message.definition_message)
            power_field = message.get_field(RecordPowerField.ID)
            if power_field is None:
                power_field = RecordPowerField()
                message.fields.append(power_field)
            power_field.growable = True
            power_field.set_value(0, power_by_record.get(record_index, 0))
            record_index += 1
        builder.add(message)

    if len(power_by_record) != len(powers):
        raise ValueError(
            "Calculated power and FIT record indexes have different lengths"
        )

    builder.build().to_file(output_path)
