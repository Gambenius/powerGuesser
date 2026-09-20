import pandas as pd
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
    df['fit_record_index'] = range(len(df))
    
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
    needed = ['fit_record_index', 'time', 'ele', 'speed', 'pos_lat', 'pos_lon', 'cad', 'hr', 'temp', 'real_power']
    df = df[[c for c in needed if c in df.columns]].copy()
    
    df = df.dropna(subset=['pos_lat']).reset_index(drop=True)

    sc_to_deg = 180 / pow(2, 31)
    df['lat'] = df['pos_lat'] * sc_to_deg
    df['lon'] = df['pos_lon'] * sc_to_deg
    
    df['dt'] = df['time'].diff().dt.total_seconds().fillna(1.0)
    df.loc[df['dt'] <= 0, 'dt'] = 1.0
    
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
