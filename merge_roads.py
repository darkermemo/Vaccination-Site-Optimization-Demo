import osmnx as ox
import networkx as nx
import os
import pickle

def merge_roads():
    print("🔄 Merging road chunks...")
    cache_dir = "cache/roads"
    full_graph_path = os.path.join(cache_dir, "riyadh_full.pkl")
    
    if os.path.exists(cache_dir):
        chunk_files = [f for f in os.listdir(cache_dir) if f.endswith(".graphml") and "chunk" in f]
        if chunk_files:
            try:
                # Load and compose chunks
                G = nx.MultiDiGraph()
                total_chunks = len(chunk_files)
                print(f"Found {total_chunks} chunks.")
                
                for i, f in enumerate(chunk_files):
                    print(f"Merging chunk {i+1}/{total_chunks}...")
                    filepath = os.path.join(cache_dir, f)
                    try:
                        G_chunk = ox.load_graphml(filepath)
                        G = nx.compose(G, G_chunk)
                    except Exception as e:
                        print(f"Error loading chunk {f}: {e}")
                
                if len(G) > 0:
                    print(f"✅ Merged graph has {len(G)} nodes.")
                    # Save merged graph (Pickle)
                    with open(full_graph_path, 'wb') as f:
                        pickle.dump(G, f)
                    print(f"💾 Saved merged graph to {full_graph_path}")
                else:
                    print("❌ Merged graph is empty.")
            except Exception as e:
                print(f"❌ Error merging chunks: {e}")
        else:
            print("❌ No chunks found.")
    else:
        print("❌ Cache directory not found.")

if __name__ == "__main__":
    merge_roads()
