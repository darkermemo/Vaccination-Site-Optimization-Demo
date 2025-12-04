import osmnx as ox
import networkx as nx
import os
import time
from shapely.geometry import box

def create_grid(gdf, n_rows=6, n_cols=6):
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

def preload_roads(place_name="Riyadh, Saudi Arabia"):
    print(f"🚀 Starting incremental road download for {place_name}...")
    
    # Create cache directory
    cache_dir = "cache/roads"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Get city boundary
    try:
        city_gdf = ox.geocode_to_gdf(place_name)
    except Exception as e:
        print(f"Error getting city boundary: {e}")
        return

    # Create grid (6x6 = 36 chunks for smaller, safer downloads)
    grid = create_grid(city_gdf, n_rows=6, n_cols=6)
    total_chunks = len(grid)
    
    print(f"ℹ️ Divided city into {total_chunks} chunks.")
    
    for i, polygon in enumerate(grid):
        filename = os.path.join(cache_dir, f"riyadh_road_chunk_{i}.graphml")
        
        if os.path.exists(filename):
            print(f"✅ Chunk {i+1}/{total_chunks} already exists. Skipping.")
            continue
            
        print(f"⏳ Downloading chunk {i+1}/{total_chunks}...")
        try:
            # Download graph for this chunk
            G_chunk = ox.graph_from_polygon(polygon, network_type='drive')
            
            if G_chunk is not None and len(G_chunk) > 0:
                # Add speeds and travel times immediately
                G_chunk = ox.add_edge_speeds(G_chunk)
                G_chunk = ox.add_edge_travel_times(G_chunk)
                
                # Save to GraphML
                ox.save_graphml(G_chunk, filename)
                print(f"✅ Saved chunk {i+1} with {len(G_chunk)} nodes.")
            else:
                print(f"⚠️ Chunk {i+1} is empty (no roads).")
                
        except Exception as e:
            print(f"❌ Error downloading chunk {i+1}: {e}")
            # Sleep a bit to be nice to API if we hit a limit
            time.sleep(5)
            
    print("🎉 All chunks processed!")

if __name__ == "__main__":
    # Configure OSMnx
    ox.settings.log_console = True
    ox.settings.use_cache = True
    preload_roads()
