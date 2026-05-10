import io
from fitparse import FitFile
from fit_tool.fitter import Fitter
import copy


def export_to_fit_with_power(input_fit_path, df):
    """
    Read a FIT file, inject the calculated power values, and return as bytes.
    Uses fit_tool library which has proper FIT file write support.
    
    Args:
        input_fit_path: Path to the original FIT file
        df: DataFrame containing the calculated 'p_guessed' power column
    
    Returns:
        bytes: The modified FIT file data
    """
    # Read the original FIT file
    fitfile = FitFile(input_fit_path)
    
    # Extract all messages from original file
    fitter = Fitter()
    
    record_index = 0
    
    # Process all messages from the original file
    for message in fitfile.messages:
        if message.name == 'record':
            # Convert record fields to a dictionary
            record_dict = {}
            for field in message.fields:
                record_dict[field.name] = field.value
            
            # Inject the calculated power
            if record_index < len(df):
                power_value = int(max(0, df.iloc[record_index]['p_guessed']))
                record_dict['power'] = power_value
            
            # Add the record to the fitter
            fitter.records.append(record_dict)
            record_index += 1
        else:
            # For non-record messages, copy them as-is
            message_dict = {}
            for field in message.fields:
                message_dict[field.name] = field.value
            
            # Store message data (we'll add file_id, device_info, etc.)
            if message.name == 'file_id':
                fitter.file_id = message_dict
            elif message.name == 'device_info':
                if not hasattr(fitter, 'devices'):
                    fitter.devices = []
                fitter.devices.append(message_dict)
            elif message.name == 'activity':
                fitter.activity = message_dict
    
    # Generate the FIT file as bytes
    output = fitter.to_bytes()
    return output


def save_fit_bytes_simple(input_fit_path, df):
    """
    Simpler approach: Read original FIT, extract records with power injection,
    and write back using fit_tool.
    
    Args:
        input_fit_path: Path to the original FIT file
        df: DataFrame containing the calculated 'p_guessed' power column
    
    Returns:
        bytes: The modified FIT file data
    """
    fitfile = FitFile(input_fit_path)
    fitter = Fitter()
    
    record_index = 0
    
    # Copy all messages with modified power in records
    for message in fitfile.messages:
        if message.name == 'record':
            record_dict = {field.name: field.value for field in message.fields}
            
            if record_index < len(df):
                record_dict['power'] = int(max(0, df.iloc[record_index]['p_guessed']))
            
            fitter.records.append(record_dict)
            record_index += 1
        else:
            msg_dict = {field.name: field.value for field in message.fields}
            
            if message.name == 'file_id':
                fitter.file_id = msg_dict
            elif message.name == 'activity':
                fitter.activity = msg_dict
            elif message.name == 'device_info':
                if not hasattr(fitter, 'devices'):
                    fitter.devices = []
                fitter.devices.append(msg_dict)
    
    return fitter.to_bytes()
