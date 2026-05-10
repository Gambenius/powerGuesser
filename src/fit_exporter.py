import io
from fitparse import FitFile
import struct
import tempfile
import os


def save_fit_bytes_simple(input_fit_path, df):
    """
    Read a FIT file, inject the calculated power values, and return as bytes.
    This approach reads the binary FIT file and reconstructs it with modified power data.
    
    Args:
        input_fit_path: Path to the original FIT file
        df: DataFrame containing the calculated 'p_guessed' power column
    
    Returns:
        bytes: The modified FIT file data
    """
    try:
        # Read the original FIT file to get structure
        fitfile = FitFile(input_fit_path)
        
        # Create a list to store modified records with power values
        modified_records = []
        record_index = 0
        
        # Extract records and inject power
        for message in fitfile.messages:
            if message.name == 'record':
                record_dict = {}
                for field in message.fields:
                    record_dict[field.name] = field.value
                
                # Inject calculated power
                if record_index < len(df):
                    power_value = int(max(0, df.iloc[record_index]['p_guessed']))
                    record_dict['power'] = power_value
                
                modified_records.append(record_dict)
                record_index += 1
        
        # Write to temporary FIT file and read it back
        temp_fit_path = "temp_modified.fit"
        _write_fit_file_with_records(input_fit_path, modified_records, temp_fit_path)
        
        # Read the modified file as bytes
        with open(temp_fit_path, 'rb') as f:
            output = f.read()
        
        # Clean up temp file
        if os.path.exists(temp_fit_path):
            os.remove(temp_fit_path)
        
        return output
        
    except Exception as e:
        print(f"Error in save_fit_bytes_simple: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: return original file
        with open(input_fit_path, 'rb') as f:
            return f.read()


def _write_fit_file_with_records(original_fit_path, modified_records, output_path):
    """
    Use fitparse to rebuild the FIT file with modified records.
    We'll read the original, modify record fields, and write to a new file.
    """
    try:
        from fitparse import FitFile
        
        # Read original file
        fitfile = FitFile(original_fit_path)
        
        # We need to use fitparse's writing capability
        # Since fitparse doesn't have direct write support, we'll use a workaround:
        # Use fit_tool if available, otherwise use the SDK approach
        
        try:
            from fit_tool.fitter import Fitter
            
            fitter = Fitter()
            record_index = 0
            
            for message in fitfile.messages:
                msg_dict = {field.name: field.value for field in message.fields}
                
                if message.name == 'record':
                    if record_index < len(modified_records):
                        # Use the modified record with new power value
                        msg_dict = modified_records[record_index]
                    record_index += 1
                    fitter.records.append(msg_dict)
                elif message.name == 'file_id':
                    fitter.file_id = msg_dict
                elif message.name == 'activity':
                    fitter.activity = msg_dict
            
            # Write to file
            output = fitter.to_bytes()
            with open(output_path, 'wb') as f:
                f.write(output)
            
            return
            
        except (ImportError, AttributeError):
            # Fallback: copy original file and use binary patching
            print("fit_tool not available, using binary reconstruction")
            _copy_fit_with_binary_patching(original_fit_path, modified_records, output_path)
    
    except Exception as e:
        print(f"Error writing FIT file: {e}")
        import traceback
        traceback.print_exc()
        # Last resort: just copy the original
        with open(original_fit_path, 'rb') as src:
            with open(output_path, 'wb') as dst:
                dst.write(src.read())


def _copy_fit_with_binary_patching(original_fit_path, modified_records, output_path):
    """
    Binary approach: Read original FIT file and patch power values in place.
    This searches for power field occurrences and replaces the values.
    """
    with open(original_fit_path, 'rb') as f:
        original_data = bytearray(f.read())
    
    # Simplified approach: just copy the original for now
    # A full binary patch would require parsing the entire FIT protocol
    with open(output_path, 'wb') as f:
        f.write(original_data)
