import requests
import time

def get_elevation_batch(df, chunk_size=100):
    """
    Queries Open-Elevation API in chunks to avoid timeouts.
    Returns a list of elevations.
    """
    elevations = []
    # Prepare coordinates
    coords = df[['lat', 'lon']].to_dict('records')
    
    for i in range(0, len(coords), chunk_size):
        chunk = coords[i:i + chunk_size]
        payload = {"locations": chunk}
        
        try:
            # Using the public API (or you can host your own local instance)
            response = requests.post("https://api.open-elevation.com/api/v1/lookup", json=payload)
            if response.status_code == 200:
                results = response.json()['results']
                elevations.extend([r['elevation'] for r in results])
            else:
                print(f"Error at chunk {i}: {response.status_code}")
                # Fallback to existing elevation if API fails
                elevations.extend(df['ele'].iloc[i:i + chunk_size].tolist())
        except Exception as e:
            print(f"Request failed: {e}")
            elevations.extend(df['ele'].iloc[i:i + chunk_size].tolist())
            
        # Be nice to the free API
        time.sleep(0.2) 
        
    return elevations