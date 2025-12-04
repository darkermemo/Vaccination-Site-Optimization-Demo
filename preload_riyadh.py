import os
import re
import pandas as pd
import osmnx as ox
import numpy as np
from shapely.geometry import box

# Configure osmnx
ox.settings.log_console = True
ox.settings.use_cache = True

def sanitize_filename(name):
    """Sanitize string to be safe for filenames"""
    return re.sub(r'[^\w\-_]', '_', name)

def create_grid(gdf, n_rows=4, n_cols=4):
    """Create a grid of polygons from a bounding box"""
    xmin, ymin, xmax, ymax = gdf.total_bounds
    width = (xmax - xmin) / n_cols
    height = (ymax - ymin) / n_rows
    
    grid_cells = []
    for i in range(n_cols):
        for j in range(n_rows):
            x0 = xmin + i * width
            y0 = ymin + j * height
            x1 = x0 + width
            y1 = y0 + height
            grid_cells.append(box(x0, y0, x1, y1))
    return grid_cells

def auto_extract_facilities(place_name, tags=None):
    """Extract facilities from OpenStreetMap or local CSV using incremental grid extraction"""
    if tags is None:
        tags = {"amenity": ["hospital", "clinic", "doctors"]}
    
    filename = f"facilities_{sanitize_filename(place_name)}.csv"
    
    print(f"⏳ Contacting OpenStreetMap for {place_name}... using incremental grid extraction.")
    
    try:
        # 1. Get city boundary
        city_gdf = ox.geocode_to_gdf(place_name)
        
        # 2. Create grid (4x4 = 16 chunks)
        grid = create_grid(city_gdf, n_rows=4, n_cols=4)
        total_chunks = len(grid)
        
        # Initialize file with header if it doesn't exist
        if not os.path.exists(filename):
            pd.DataFrame(columns=["name", "latitude", "longitude"]).to_csv(filename, index=False)
        
        count = 0
        for i, polygon in enumerate(grid):
            print(f"Processing chunk {i+1}/{total_chunks}...")
            try:
                # Extract from this chunk
                chunk_gdf = ox.features_from_polygon(polygon, tags)
                
                if len(chunk_gdf) > 0:
                    chunk_gdf["latlong"] = chunk_gdf.geometry.centroid
                    chunk_gdf["latitude"] = chunk_gdf.latlong.y
                    chunk_gdf["longitude"] = chunk_gdf.latlong.x
                    
                    # Extract relevant columns
                    chunk_result = chunk_gdf[["name", "latitude", "longitude"]].dropna()
                    chunk_result = chunk_result[chunk_result['name'].notna()]
                    
                    if len(chunk_result) > 0:
                        # Append to CSV immediately
                        chunk_result.to_csv(filename, mode='a', header=False, index=False)
                        count += len(chunk_result)
                        print(f"   -> Found {len(chunk_result)} items")
            except Exception as e:
                pass
            
        print("Finalizing data...")
        
        # Load full file, remove duplicates, and save clean version
        if os.path.exists(filename):
            full_df = pd.read_csv(filename)
            full_df = full_df.drop_duplicates(subset=['name', 'latitude', 'longitude'])
            full_df.to_csv(filename, index=False)
            print(f"✅ Found {len(full_df)} facilities! Saved to {filename}")
        else:
            print("No facilities found in any chunk.")

    except Exception as e:
        print(f"Error: {e}")

def auto_extract_districts(place_name):
    """Extract districts/neighborhoods from OpenStreetMap or local CSV using incremental grid extraction"""
    
    filename = f"districts_{sanitize_filename(place_name)}.csv"
    
    print(f"⏳ Contacting OpenStreetMap for {place_name}... using incremental grid extraction.")
    
    try:
        # 1. Get city boundary
        city_gdf = ox.geocode_to_gdf(place_name)
        
        # 2. Create grid (4x4 = 16 chunks)
        grid = create_grid(city_gdf, n_rows=4, n_cols=4)
        total_chunks = len(grid)
        
        # Initialize file with header if it doesn't exist
        if not os.path.exists(filename):
            pd.DataFrame(columns=['Village_name', 'population', 'infected', 'latitude', 'longitude']).to_csv(filename, index=False)
        
        count = 0
        tags = {"admin_level": ["9", "10"], "place": ["neighbourhood", "suburb", "quarter"]}
        
        for i, polygon in enumerate(grid):
            print(f"Processing chunk {i+1}/{total_chunks}...")
            try:
                # Extract from this chunk
                chunk_gdf = ox.features_from_polygon(polygon, tags)
                
                if len(chunk_gdf) > 0:
                    chunk_gdf["latlong"] = chunk_gdf.geometry.centroid
                    chunk_gdf["latitude"] = chunk_gdf.latlong.y
                    chunk_gdf["longitude"] = chunk_gdf.latlong.x
                    
                    # Extract name column
                    if 'name' in chunk_gdf.columns:
                        chunk_result = chunk_gdf[["name", "latitude", "longitude"]].dropna()
                    elif 'name:en' in chunk_gdf.columns:
                        chunk_result = chunk_gdf[["name:en", "latitude", "longitude"]].dropna()
                        chunk_result = chunk_result.rename(columns={'name:en': 'name'})
                    else:
                        continue
                    
                    if len(chunk_result) > 0:
                        # Generate random but realistic population and infection data
                        np.random.seed(42 + i)
                        chunk_result['Village_name'] = chunk_result['name']
                        chunk_result['population'] = np.random.randint(5000, 150000, size=len(chunk_result))
                        chunk_result['infected'] = (chunk_result['population'] * np.random.uniform(0.005, 0.025, size=len(chunk_result))).astype(int)
                        
                        chunk_result = chunk_result[['Village_name', 'population', 'infected', 'latitude', 'longitude']]
                        
                        # Append to CSV immediately
                        chunk_result.to_csv(filename, mode='a', header=False, index=False)
                        count += len(chunk_result)
                        print(f"   -> Found {len(chunk_result)} items")
            except Exception as e:
                pass
            
        print("Finalizing data...")
        
        # Load full file, remove duplicates, and save clean version
        if os.path.exists(filename):
            full_df = pd.read_csv(filename)
            full_df = full_df.drop_duplicates(subset=['Village_name'])
            full_df.to_csv(filename, index=False)
            print(f"✅ Generated data for {len(full_df)} districts! Saved to {filename}")
        else:
            print("No districts found.")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    place = "Riyadh, Saudi Arabia"
    print(f"🚀 Starting preload for {place}...")
    
    print("\n--- Extracting Facilities ---")
    auto_extract_facilities(place)
    
    print("\n--- Extracting Districts ---")
    auto_extract_districts(place)
    
    print("\n✨ Preload complete!")
