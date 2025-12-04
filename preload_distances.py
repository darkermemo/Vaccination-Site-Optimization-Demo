import pandas as pd
import networkx as nx
import osmnx as ox
import numpy as np
import pickle
import os
import re

def sanitize_filename(name):
    return re.sub(r'[^\w\s-]', '', name).replace(' ', '_')

def preload_distances():
    place_name = "Riyadh, Saudi Arabia"
    print(f"🚀 Starting background distance calculation for {place_name}...")
    
    # Load Data
    vacc = pd.read_csv("facilities_Riyadh__Saudi_Arabia.csv")
    vill = pd.read_csv("districts_Riyadh__Saudi_Arabia.csv")
    
    print(f"Loaded {len(vacc)} facilities and {len(vill)} districts.")
    
    # Load Graph
    cache_dir = "cache/roads"
    full_graph_path = os.path.join(cache_dir, "riyadh_full.pkl")
    
    if not os.path.exists(full_graph_path):
        print("❌ Road graph not found! Please wait for preload_roads.py to finish.")
        return
        
    print("⏳ Loading road graph...")
    with open(full_graph_path, 'rb') as f:
        G = pickle.load(f)
    print(f"✅ Loaded graph with {len(G)} nodes.")
    
    # Prepare DataFrame
    # Handle case sensitivity for 'name' column
    if 'Name' in vacc.columns:
        index = vacc.Name
    elif 'name' in vacc.columns:
        index = vacc.name
    else:
        index = vacc.iloc[:, 0]
        
    columns = vill.Village_name
    df_distances = pd.DataFrame(index=index, columns=columns)
    
    # Calculate Distances
    print("Mapping locations to nodes...")
    vacc_nodes = ox.nearest_nodes(G, Y=vacc.latitude, X=vacc.longitude)
    vill_nodes = ox.nearest_nodes(G, Y=vill.latitude, X=vill.longitude)
    
    total_pairs = len(vacc) * len(vill)
    print(f"Computing {total_pairs} shortest paths...")
    
    completed = 0
    for i in range(len(vacc)):
        for j in range(len(vill)):
            try:
                origin_node = vill_nodes[j]
                destination_node = vacc_nodes[i]
                dist = nx.shortest_path_length(G, origin_node, destination_node, weight='length')
                df_distances.iloc[i,j] = dist
            except:
                # Fallback
                dist = np.sqrt((vacc.iloc[i].latitude - vill.iloc[j].latitude)**2 + (vacc.iloc[i].longitude - vill.iloc[j].longitude)**2) * 111000
                df_distances.iloc[i,j] = dist
            
            completed += 1
            if completed % 100 == 0: # More frequent updates
                print(f"Progress: {completed}/{total_pairs} ({completed/total_pairs*100:.1f}%)", flush=True)
    
    # Save Cache
    cache_filename = f"cache/distances_{sanitize_filename(place_name)}_{len(vacc)}_{len(vill)}_road.pkl"
    os.makedirs("cache", exist_ok=True)
    
    with open(cache_filename, 'wb') as f:
        pickle.dump(df_distances, f)
        
    print(f"🎉 Done! Saved distance matrix to {cache_filename}", flush=True)

if __name__ == "__main__":
    preload_distances()
