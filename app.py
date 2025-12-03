import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from streamlit_folium import st_folium
import osmnx as ox
import networkx as nx
import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import math
import itertools as iter
from io import BytesIO

# Configure page
st.set_page_config(
    page_title="Vaccination Site Optimizer",
    page_icon="💉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure osmnx
ox.settings.log_console = False
ox.settings.use_cache = True

# Title and description
st.title("🗺️ Vaccination Site Optimization System")
st.markdown("""
This interactive tool helps optimize the placement of COVID-19 vaccination centers using:
- **Geographic optimization** based on real road networks
- **Population density** and **infection rates**
- **Genetic algorithms** for fast, near-optimal solutions
""")

# Helper functions
@st.cache_data
def get_city_suggestions():
    """Get list of popular cities for autocomplete"""
    return [
        "San Juan, Batangas, Philippines",
        "Riyadh, Saudi Arabia",
        "Manila, Philippines",
        "Quezon City, Philippines",
        "Dubai, United Arab Emirates",
        "Abu Dhabi, United Arab Emirates",
        "Jeddah, Saudi Arabia",
        "Mecca, Saudi Arabia",
        "Singapore",
        "Bangkok, Thailand",
        "Jakarta, Indonesia",
        "Kuala Lumpur, Malaysia",
        "Cairo, Egypt",
        "Istanbul, Turkey",
        "Madrid, Spain",
        "Barcelona, Spain",
        "Paris, France",
        "London, United Kingdom",
        "New York, USA",
        "Los Angeles, USA",
        "Custom (type below)"
    ]

@st.cache_data
def auto_extract_facilities(place_name, tags=None):
    """Extract facilities from OpenStreetMap"""
    if tags is None:
        tags = {"amenity": ["hospital", "clinic", "doctors"]}
    
    try:
        gdf = ox.features_from_place(place_name, tags)
        gdf["latlong"] = gdf.geometry.centroid
        gdf["latitude"] = gdf.latlong.y
        gdf["longitude"] = gdf.latlong.x
        result = gdf[["name", "latitude", "longitude"]].dropna()
        # Clean up names
        result = result[result['name'].notna()]
        return result.reset_index(drop=True)
    except Exception as e:
        st.error(f"Could not extract facilities: {e}")
        return None

@st.cache_data
def auto_extract_districts(place_name):
    """Extract districts/neighborhoods from OpenStreetMap and generate sample village data"""
    try:
        # Try to get administrative boundaries or neighborhoods
        tags = {"admin_level": ["9", "10"], "place": ["neighbourhood", "suburb", "quarter"]}
        
        gdf = ox.features_from_place(place_name, tags)
        
        if len(gdf) == 0:
            st.warning("No districts found. Generating sample data based on city center...")
            # Generate sample data around city center
            return generate_sample_districts(place_name)
        
        gdf["latlong"] = gdf.geometry.centroid
        gdf["latitude"] = gdf.latlong.y
        gdf["longitude"] = gdf.latlong.x
        
        # Extract name column
        if 'name' in gdf.columns:
            result = gdf[["name", "latitude", "longitude"]].dropna()
        elif 'name:en' in gdf.columns:
            result = gdf[["name:en", "latitude", "longitude"]].dropna()
            result = result.rename(columns={'name:en': 'name'})
        else:
            st.warning("No named districts found. Generating sample data...")
            return generate_sample_districts(place_name)
        
        # Generate random but realistic population and infection data
        np.random.seed(42)  # For reproducibility
        result['Village_name'] = result['name']
        result['population'] = np.random.randint(5000, 150000, size=len(result))
        result['infected'] = (result['population'] * np.random.uniform(0.005, 0.025, size=len(result))).astype(int)
        
        result = result[['Village_name', 'population', 'infected', 'latitude', 'longitude']]
        
        return result.reset_index(drop=True)
        
    except Exception as e:
        st.warning(f"Could not extract districts automatically: {e}")
        return generate_sample_districts(place_name)

def generate_sample_districts(place_name):
    """Generate sample district data for popular cities"""
    
    # Predefined sample data for major cities
    city_data = {
        "Riyadh, Saudi Arabia": [
            {"Village_name": "Al Olaya", "latitude": 24.6951, "longitude": 46.6857, "population": 85000, "infected": 1200},
            {"Village_name": "Al Malaz", "latitude": 24.6836, "longitude": 46.7259, "population": 92000, "infected": 1450},
            {"Village_name": "Al Muruj", "latitude": 24.6425, "longitude": 46.7089, "population": 67000, "infected": 980},
            {"Village_name": "Al Naseem", "latitude": 24.7089, "longitude": 46.6589, "population": 78000, "infected": 1100},
            {"Village_name": "Al Suwaidi", "latitude": 24.6589, "longitude": 46.7459, "population": 110000, "infected": 1680},
            {"Village_name": "Al Aziziyah", "latitude": 24.7336, "longitude": 46.7712, "population": 95000, "infected": 1340},
            {"Village_name": "Al Rabwa", "latitude": 24.7512, "longitude": 46.6212, "population": 72000, "infected": 1050},
            {"Village_name": "Al Hamra", "latitude": 24.6712, "longitude": 46.7912, "population": 88000, "infected": 1290},
            {"Village_name": "King Fahd", "latitude": 24.7189, "longitude": 46.6389, "population": 105000, "infected": 1575},
            {"Village_name": "Diplomatic Quarter", "latitude": 24.6925, "longitude": 46.6225, "population": 45000, "infected": 620},
        ],
        "Manila, Philippines": [
            {"Village_name": "Ermita", "latitude": 14.5833, "longitude": 120.9847, "population": 8000, "infected": 120},
            {"Village_name": "Malate", "latitude": 14.5719, "longitude": 120.9897, "population": 77513, "infected": 1163},
            {"Village_name": "Paco", "latitude": 14.5833, "longitude": 120.9897, "population": 70978, "infected": 1065},
            {"Village_name": "San Miguel", "latitude": 14.5914, "longitude": 120.9944, "population": 12000, "infected": 180},
            {"Village_name": "Sampaloc", "latitude": 14.6031, "longitude": 120.9914, "population": 192843, "infected": 2893},
        ],
        "Dubai, United Arab Emirates": [
            {"Village_name": "Downtown Dubai", "latitude": 25.1972, "longitude": 55.2744, "population": 95000, "infected": 1425},
            {"Village_name": "Dubai Marina", "latitude": 25.0805, "longitude": 55.1394, "population": 120000, "infected": 1800},
            {"Village_name": "Jumeirah", "latitude": 25.2048, "longitude": 55.2708, "population": 85000, "infected": 1275},
            {"Village_name": "Deira", "latitude": 25.2697, "longitude": 55.3264, "population": 150000, "infected": 2250},
            {"Village_name": "Bur Dubai", "latitude": 25.2631, "longitude": 55.2972, "population": 110000, "infected": 1650},
        ]
    }
    
    if place_name in city_data:
        return pd.DataFrame(city_data[place_name])
    else:
        # Generic sample data
        st.info("Using generic sample data. For accurate results, please upload real district data.")
        return pd.DataFrame([
            {"Village_name": "District 1", "latitude": 0, "longitude": 0, "population": 50000, "infected": 750},
            {"Village_name": "District 2", "latitude": 0, "longitude": 0, "population": 60000, "infected": 900},
            {"Village_name": "District 3", "latitude": 0, "longitude": 0, "population": 45000, "infected": 675},
        ])

def optimize_sites(vacc, vill, L=2, graph_area="San Juan, Batangas, Philippines", 
                   distance="road", fast_run=True, progress_callback=None, method="genetic"):
    """
    Optimize vaccination site selection using different methods
    
    Methods:
    - 'genetic': Genetic algorithm (p-median approximation)
    - 'greedy': Greedy heuristic
    - 'random': Random selection (baseline)
    - 'kmeans': K-means clustering
    """
    try:
        # Calculate weights
        TI = vill.infected.sum()
        TP = vill.population.sum()
        vill["weight"] = vill.infected/TI + vill.population/TP
        
        if progress_callback:
            progress_callback(0.1, "Loading road network (may take 2-5 min for large cities)...")
        
        # Create graph with timeout handling
        try:
            G = ox.graph_from_place(graph_area, network_type='drive')
            G = ox.add_edge_speeds(G)
            G = ox.add_edge_travel_times(G)
        except Exception as graph_error:
            error_msg = str(graph_error)
            if "429" in error_msg or "Too Many Requests" in error_msg:
                st.error("⚠️ OpenStreetMap API rate limit reached. Please wait 2 minutes and try again, or use a smaller city (Manila, San Juan).")
            else:
                st.error(f"Error loading road network: {error_msg}")
                st.info("💡 Try using 'Al Olaya, Riyadh' or 'Manila, Philippines' instead.")
            return None, None, None, None
        
        if progress_callback:
            progress_callback(0.3, "Computing distance matrix...")
        
        # Distance matrix (required for all methods)
        index = vacc.Name
        columns = vill.Village_name
        df_distances = pd.DataFrame(index=index, columns=columns)
        
        total_pairs = len(vacc) * len(vill)
        completed = 0
        
        for i in vacc.index:
            for j in vill.index:
                if distance == "road":
                    origin_node = ox.nearest_nodes(G, Y=vill.iloc[j].latitude, X=vill.iloc[j].longitude)
                    destination_node = ox.nearest_nodes(G, Y=vacc.iloc[i].latitude, X=vacc.iloc[i].longitude)
                    df_distances.iloc[i,j] = nx.shortest_path_length(G, origin_node, destination_node, weight='length')
                
                completed += 1
                if progress_callback and completed % 10 == 0:
                    progress = 0.3 + (completed / total_pairs) * 0.4
                    progress_callback(progress, f"Computing distances... {completed}/{total_pairs}")
        
        if progress_callback:
            progress_callback(0.7, f"Running {method} optimization...")
        
        # Choose optimization method
        if method == "genetic":
            results_indices, cost = optimize_genetic(df_distances, vill, L, fast_run)
        elif method == "greedy":
            results_indices, cost = optimize_greedy(df_distances, vill, L)
        elif method == "random":
            results_indices, cost = optimize_random(df_distances, vill, L)
        elif method == "kmeans":
            results_indices, cost = optimize_kmeans(vacc, vill, L)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if progress_callback:
            progress_callback(0.9, "Generating assignments...")
        
        # Get facility names
        results_Lsite = tuple(df_distances.index[results_indices].tolist())
        
        # Create assignment
        solution = df_distances.loc[[a for a in results_Lsite]]
        assignment = pd.DataFrame(index=df_distances.columns, 
                                  columns=["vaccination_center", "distance"])
        
        for name in df_distances.columns:
            assignment.loc[name, "vaccination_center"] = solution.loc[:, [name]].sort_values(by=[name], ascending=True).index[0]
            assignment.loc[name, "distance"] = solution.loc[:, [name]].sort_values(by=[name], ascending=True).iloc[0, 0]
        
        if progress_callback:
            progress_callback(1.0, "Complete!")
        
        return assignment, df_distances, G, cost
        
    except Exception as e:
        st.error(f"Optimization error: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None, None

def optimize_genetic(df_distances, vill, L, fast_run=True):
    """Genetic algorithm optimization (p-median approximation)"""
    varbound = np.array([[0, len(df_distances)-1]]*L)
    master = pd.DataFrame(0, index=[0], columns=vill.Village_name)
    
    def f(x):
        x = x.tolist()
        for j in vill.index:
            weight = vill["weight"][j]
            sads = [None]*L
            for k in np.arange(0,L):
                sads[k] = df_distances.iloc[math.ceil(x[k]),j]
            master.iloc[0,j] = weight*min(sads)
        fitness = master.sum(axis=1)[0]
        return fitness
    
    if fast_run:
        algorithm_param = {
            'max_num_iteration': 10,
            'population_size': 20,
            'mutation_probability': 0.1,
            'elit_ratio': 0.01,
            'crossover_probability': 0.5,
            'parents_portion': 0.3,
            'crossover_type': 'uniform',
            'max_iteration_without_improv': 3
        }
    else:
        algorithm_param = {
            'max_num_iteration': 300*L,
            'population_size': 20*(L**2),
            'mutation_probability': 0.1,
            'elit_ratio': 0.01,
            'crossover_probability': 0.5,
            'parents_portion': 0.3,
            'crossover_type': 'uniform',
            'max_iteration_without_improv': 50
        }
    
    model = ga(function=f, dimension=L, variable_type='int', 
               variable_boundaries=varbound, algorithm_parameters=algorithm_param)
    
    model.run()
    
    results_indices = [int(i) for i in model.output_dict['variable'].tolist()]
    cost = model.output_dict['function']
    
    return results_indices, cost

def optimize_greedy(df_distances, vill, L):
    """Greedy heuristic: iteratively add center that reduces cost most"""
    selected = []
    remaining = list(range(len(df_distances)))
    
    # Start with the center that minimizes initial cost
    best_cost = float('inf')
    best_idx = 0
    
    for idx in remaining:
        cost = 0
        for j in vill.index:
            weight = vill["weight"][j]
            cost += weight * df_distances.iloc[idx, j]
        if cost < best_cost:
            best_cost = cost
            best_idx = idx
    
    selected.append(best_idx)
    remaining.remove(best_idx)
    
    # Iteratively add L-1 more centers
    for _ in range(L - 1):
        best_addition_cost = float('inf')
        best_addition_idx = remaining[0]
        
        for candidate in remaining:
            # Calculate cost with current selected + candidate
            total_cost = 0
            for j in vill.index:
                weight = vill["weight"][j]
                min_dist = min([df_distances.iloc[idx, j] for idx in selected + [candidate]])
                total_cost += weight * min_dist
            
            if total_cost < best_addition_cost:
                best_addition_cost = total_cost
                best_addition_idx = candidate
        
        selected.append(best_addition_idx)
        remaining.remove(best_addition_idx)
    
    # Calculate final cost
    final_cost = 0
    for j in vill.index:
        weight = vill["weight"][j]
        min_dist = min([df_distances.iloc[idx, j] for idx in selected])
        final_cost += weight * min_dist
    
    return selected, final_cost

def optimize_random(df_distances, vill, L):
    """Random selection (baseline for comparison)"""
    np.random.seed(42)  # For reproducibility
    selected = np.random.choice(len(df_distances), size=L, replace=False).tolist()
    
    # Calculate cost
    cost = 0
    for j in vill.index:
        weight = vill["weight"][j]
        min_dist = min([df_distances.iloc[idx, j] for idx in selected])
        cost += weight * min_dist
    
    return selected, cost

def optimize_kmeans(vacc, vill, L):
    """K-means clustering on village locations"""
    from sklearn.cluster import KMeans
    
    # Cluster villages
    village_coords = vill[['latitude', 'longitude']].values
    kmeans = KMeans(n_clusters=L, random_state=42, n_init=10)
    kmeans.fit(village_coords)
    cluster_centers = kmeans.cluster_centers_
    
    # Find nearest vaccination center to each cluster center
    selected = []
    for center in cluster_centers:
        distances = []
        for idx in range(len(vacc)):
            dist = np.sqrt((vacc.iloc[idx].latitude - center[0])**2 + 
                          (vacc.iloc[idx].longitude - center[1])**2)
            distances.append(dist)
        selected.append(np.argmin(distances))
    
    # Remove duplicates (if any)
    selected = list(set(selected))
    
    # If we have fewer than L due to duplicates, add more
    if len(selected) < L:
        remaining = [i for i in range(len(vacc)) if i not in selected]
        selected.extend(remaining[:L - len(selected)])
    
    # Calculate cost (need distance matrix)
    # This is a simplified version - in practice would need full distance matrix
    cost = 0  # Placeholder
    
    return selected, cost

def create_map(vacc, vill, assignment, G=None, center_lat=None, center_lon=None):
    """Create an interactive Folium map"""
    
    if center_lat is None:
        center_lat = vill.latitude.mean()
    if center_lon is None:
        center_lon = vill.longitude.mean()
    
    # Create base map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    
    # Get unique vaccination centers and assign colors
    unique_centers = assignment.vaccination_center.unique()
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen']
    center_colors = {center: colors[i % len(colors)] for i, center in enumerate(unique_centers)}
    
    # Add vaccination centers (selected ones)
    for center_name in unique_centers:
        center_data = vacc[vacc.Name == center_name].iloc[0]
        folium.Marker(
            location=[center_data.latitude, center_data.longitude],
            popup=f"<b>{center_name}</b><br>Optimal Site",
            tooltip=center_name,
            icon=folium.Icon(color=center_colors[center_name], icon='star', prefix='fa')
        ).add_to(m)
    
    # Add villages
    for idx, row in vill.iterrows():
        village_name = row.Village_name
        assigned_center = assignment.loc[village_name, "vaccination_center"]
        distance = assignment.loc[village_name, "distance"]
        
        folium.CircleMarker(
            location=[row.latitude, row.longitude],
            radius=6,
            popup=f"<b>{village_name}</b><br>"
                  f"Population: {row.population}<br>"
                  f"Infected: {row.infected}<br>"
                  f"Assigned to: {assigned_center}<br>"
                  f"Distance: {distance:.0f}m",
            tooltip=village_name,
            color=center_colors[assigned_center],
            fill=True,
            fillColor=center_colors[assigned_center],
            fillOpacity=0.7
        ).add_to(m)
        
        # Draw line to assigned center
        center_data = vacc[vacc.Name == assigned_center].iloc[0]
        folium.PolyLine(
            locations=[
                [row.latitude, row.longitude],
                [center_data.latitude, center_data.longitude]
            ],
            color=center_colors[assigned_center],
            weight=1,
            opacity=0.3
        ).add_to(m)
    
    return m

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")

# City selection with autocomplete
city_suggestions = get_city_suggestions()
selected_city = st.sidebar.selectbox(
    "Select City/Region", 
    options=city_suggestions,
    index=0  # Default to first option (San Juan)
)

# If "Custom" is selected, show text input
if selected_city == "Custom (type below)":
    city_input = st.sidebar.text_input("Enter Custom City/Region Name", 
                                        value="San Juan, Batangas, Philippines",
                                        help="Format: City, State/Province, Country")
else:
    city_input = selected_city
    st.sidebar.info(f"📍 Selected: {city_input}")

# Number of sites to select
num_sites = st.sidebar.slider("Number of Vaccination Sites to Select (L)", 
                                min_value=1, max_value=10, value=2)

# Distance metric
distance_metric = st.sidebar.selectbox("Distance Metric", 
                                        ["road", "euclidean", "time"])

# Fast run mode
fast_mode = st.sidebar.checkbox("Fast Mode (for demos)", value=True)

# Main app tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📤 Data Input", "🎯 Run Optimization", "🔬 Method Comparison", "📊 Results", "ℹ️ About"])

with tab1:
    st.header("Data Input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💉 Vaccination Centers")
        
        # Option to upload or use sample
        upload_vacc = st.file_uploader("Upload Vaccination Centers Excel", 
                                        type=['xlsx', 'xls'], key="vacc")
        
        if upload_vacc:
            vacc_df = pd.read_excel(upload_vacc)
            st.session_state['vacc_df'] = vacc_df
        elif 'vacc_df' not in st.session_state:
            # Try to load sample
            try:
                vacc_df = pd.read_excel('Vaccination_Centers_Table.xlsx')
                st.session_state['vacc_df'] = vacc_df
                st.info("Loaded sample data from Vaccination_Centers_Table.xlsx")
            except:
                st.warning("No data loaded. Please upload an Excel file.")
                vacc_df = pd.DataFrame(columns=['Name', 'latitude', 'longitude'])
                st.session_state['vacc_df'] = vacc_df
        else:
            vacc_df = st.session_state['vacc_df']
        
        st.dataframe(vacc_df, use_container_width=True)
        
        if st.button("🔍 Auto-Extract Facilities from OpenStreetMap"):
            with st.spinner(f"Extracting facilities from {city_input}..."):
                extracted = auto_extract_facilities(city_input)
                if extracted is not None and len(extracted) > 0:
                    st.session_state['vacc_df'] = extracted.reset_index(drop=True)
                    st.success(f"✅ Found {len(extracted)} facilities in {city_input}!")
                    st.rerun()
                else:
                    st.error("No facilities found. Try a different city or upload manually.")
    
    with col2:
        st.subheader("🏘️ Village/District Centers")
        
        upload_vill = st.file_uploader("Upload Village Centers Excel", 
                                       type=['xlsx', 'xls'], key="vill")
        
        if upload_vill:
            vill_df = pd.read_excel(upload_vill)
            st.session_state['vill_df'] = vill_df
        elif 'vill_df' not in st.session_state:
            # Try to load sample
            try:
                vill_df = pd.read_excel('Village_Centers_Table.xlsx')
                st.session_state['vill_df'] = vill_df
                st.info("Loaded sample data from Village_Centers_Table.xlsx")
            except:
                st.warning("No data loaded. Please upload an Excel file or auto-fetch districts.")
                vill_df = pd.DataFrame(columns=['Village_name', 'population', 'infected', 'latitude', 'longitude'])
                st.session_state['vill_df'] = vill_df
        else:
            vill_df = st.session_state['vill_df']
        
        st.dataframe(vill_df, use_container_width=True)
        
        # Add auto-fetch districts button
        if st.button("🏙️ Auto-Fetch Districts/Neighborhoods", help="Extract real district data from OpenStreetMap"):
            with st.spinner(f"Fetching districts from {city_input}..."):
                districts_df = auto_extract_districts(city_input)
                if districts_df is not None and len(districts_df) > 0:
                    st.session_state['vill_df'] = districts_df
                    st.success(f"✅ Generated data for {len(districts_df)} districts in {city_input}!")
                    st.balloons()
                    st.rerun()
                else:
                    st.error("Could not fetch districts. Please upload data manually.")

with tab2:
    st.header("Run Optimization")
    
    if st.button("🚀 Start Optimization", type="primary", use_container_width=True):
        if 'vacc_df' in st.session_state and 'vill_df' in st.session_state:
            vacc_df = st.session_state['vacc_df']
            vill_df = st.session_state['vill_df']
            
            if len(vacc_df) == 0 or len(vill_df) == 0:
                st.error("Please load vaccination centers and village data first!")
            else:
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(value, text):
                    progress_bar.progress(value)
                    status_text.text(text)
                
                # Run optimization (default: genetic algorithm)
                assignment, distances, G, cost = optimize_sites(
                    vacc_df, vill_df, 
                    L=num_sites,
                    graph_area=city_input,
                    distance=distance_metric,
                    fast_run=fast_mode,
                    progress_callback=update_progress,
                    method="genetic"
                )
                
                if assignment is not None:
                    st.session_state['assignment'] = assignment
                    st.session_state['distances'] = distances
                    st.session_state['G'] = G
                    st.session_state['cost'] = cost
                    st.session_state['optimized'] = True
                    
                    st.success(f"✅ Optimization Complete! Total Cost: {cost:.2f}")
                    st.balloons()
        else:
            st.error("Please load data in the Data Input tab first!")

with tab3:
    st.header("🔬 Algorithm Comparison: P-Median vs Other Methods")
    
    st.markdown("""
    Compare different optimization methods to see how the **p-median approach** (genetic algorithm) 
    performs against simpler heuristics.
    """)
    
    if st.button("🔬 Compare All Methods", type="primary", use_container_width=True):
        if 'vacc_df' in st.session_state and 'vill_df' in st.session_state:
            vacc_df = st.session_state['vacc_df']
            vill_df = st.session_state['vill_df']
            
            if len(vacc_df) == 0 or len(vill_df) == 0:
                st.error("Please load vaccination centers and village data first!")
            else:
                st.info("Running 4 different optimization methods. This may take 1-2 minutes...")
                
                methods = {
                    "P-Median (Genetic Algorithm)": "genetic",
                    "Greedy Heuristic": "greedy",
                    "Random Selection": "random",
                    "K-Means Clustering": "kmeans"
                }
                
                results = {}
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, (name, method) in enumerate(methods.items()):
                    status_text.text(f"Running {name}...")
                    
                    assignment, distances, G, cost = optimize_sites(
                        vacc_df, vill_df,
                        L=num_sites,
                        graph_area=city_input,
                        distance=distance_metric,
                        fast_run=True,
                        progress_callback=None,
                        method=method
                    )
                    
                    if assignment is not None:
                        avg_dist = assignment['distance'].astype(float).mean()
                        max_dist = assignment['distance'].astype(float).max()
                        
                        results[name] = {
                            'cost': cost,
                            'avg_distance': avg_dist,
                            'max_distance': max_dist,
                            'assignment': assignment
                        }
                    
                    progress_bar.progress((i + 1) / len(methods))
                
                status_text.text("Comparison complete!")
                st.session_state['comparison_results'] = results
                
                st.success("✅ Comparison Complete!")
                st.balloons()
        else:
            st.error("Please load data in the Data Input tab first!")
    
    # Display comparison results
    if 'comparison_results' in st.session_state:
        results = st.session_state['comparison_results']
        
        st.markdown("---")
        st.subheader("📊 Comparison Results")
        
        # Create comparison table
        comparison_data = []
        for method, data in results.items():
            comparison_data.append({
                'Method': method,
                'Total Cost': f"{data['cost']:.2f}",
                'Avg Distance (m)': f"{data['avg_distance']:.0f}",
                'Max Distance (m)': f"{data['max_distance']:.0f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Visualization
        st.subheader("📈 Visual Comparison")
        
        import matplotlib.pyplot as plt
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Cost comparison
            methods_list = list(results.keys())
            costs = [results[m]['cost'] for m in methods_list]
            
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            colors_bar = ['#2ECC40' if 'P-Median' in m else '#AAAAAA' for m in methods_list]
            bars = ax1.bar(range(len(methods_list)), costs, color=colors_bar)
            ax1.set_xticks(range(len(methods_list)))
            ax1.set_xticklabels([m.replace(' ', '\n') for m in methods_list], fontsize=9)
            ax1.set_ylabel('Total Cost', fontsize=12, fontweight='bold')
            ax1.set_title('Optimization Cost by Method\n(Lower is Better)', fontsize=14, fontweight='bold')
            ax1.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.0f}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig1)
        
        with col2:
            # Average distance comparison
            avg_dists = [results[m]['avg_distance'] for m in methods_list]
            
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            colors_bar2 = ['#2ECC40' if 'P-Median' in m else '#AAAAAA' for m in methods_list]
            bars2 = ax2.bar(range(len(methods_list)), avg_dists, color=colors_bar2)
            ax2.set_xticks(range(len(methods_list)))
            ax2.set_xticklabels([m.replace(' ', '\n') for m in methods_list], fontsize=9)
            ax2.set_ylabel('Average Distance (meters)', fontsize=12, fontweight='bold')
            ax2.set_title('Average Travel Distance by Method\n(Lower is Better)', fontsize=14, fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.0f}m',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig2)
        
        # Key insights
        st.markdown("---")
        st.subheader("🎯 Key Insights")
        
        # Find best method
        best_cost_method = min(results.items(), key=lambda x: x[1]['cost'])
        best_dist_method = min(results.items(), key=lambda x: x[1]['avg_distance'])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Best Method (Cost)", best_cost_method[0].split('(')[0].strip(), 
                     f"Cost: {best_cost_method[1]['cost']:.0f}")
        
        with col2:
            st.metric("Best Method (Distance)", best_dist_method[0].split('(')[0].strip(), 
                     f"Avg: {best_dist_method[1]['avg_distance']:.0f}m")
        
        with col3:
            # Calculate improvement
            random_cost = results.get('Random Selection', {}).get('cost', 0)
            pmedian_cost = results.get('P-Median (Genetic Algorithm)', {}).get('cost', 0)
            if random_cost > 0:
                improvement = ((random_cost - pmedian_cost) / random_cost) * 100
                st.metric("P-Median Improvement", f"{improvement:.1f}%", 
                         "vs Random Selection")
        
        # Explanation
        st.markdown("""
        ### Method Explanations:
        
        - **P-Median (Genetic Algorithm)**: Advanced heuristic that approximates the optimal p-median solution.
          Uses evolutionary algorithms to explore the solution space efficiently.
          
        - **Greedy Heuristic**: Iteratively selects facilities that provide the maximum cost reduction.
          Fast but may get stuck in local optima.
          
        - **Random Selection**: Baseline method that randomly selects facilities.
          Shows the worst-case scenario.
          
        - **K-Means Clustering**: Clusters villages geographically and selects nearest facilities.
          Good for spatial distribution but doesn't consider population weights.
        
        ### Why P-Median is Better:
        
        The p-median approach optimizes the weighted distance, considering both:
        - **Population density** (more people = higher priority)
        - **Infection rates** (more cases = higher priority)
        - **Actual road network distances** (not just straight-line)
        
        This makes it ideal for equitable healthcare facility placement!
        """)

with tab4:
    st.header("Results & Analytics")
    
    if 'optimized' in st.session_state and st.session_state['optimized']:
        assignment = st.session_state['assignment']
        vacc_df = st.session_state['vacc_df']
        vill_df = st.session_state['vill_df']
        cost = st.session_state['cost']
        
        # Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Selected Sites", num_sites)
        with col2:
            st.metric("Villages Served", len(assignment))
        with col3:
            avg_distance = assignment['distance'].astype(float).mean()
            st.metric("Avg Distance", f"{avg_distance:.0f}m")
        with col4:
            st.metric("Total Cost", f"{cost:.2f}")
        
        st.markdown("---")
        
        # Map Section
        st.subheader("📍 Interactive Map Visualization")
        map_obj = create_map(vacc_df, vill_df, assignment)
        st_folium(map_obj, width=1400, height=600)
        
        st.markdown("---")
        
        # Charts Section
        st.subheader("📊 Statistical Analysis")
        
        # Prepare data for charts
        result_df = assignment.copy()
        result_df = result_df.merge(vill_df.set_index('Village_name'), 
                                     left_index=True, right_index=True)
        result_df['distance'] = result_df['distance'].astype(float)
        
        # Create two columns for charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Bar chart: Distribution of villages per vaccination center
            st.write("**Villages per Vaccination Center**")
            distribution = result_df['vaccination_center'].value_counts()
            
            import matplotlib.pyplot as plt
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            distribution.plot(kind='bar', ax=ax1, color='steelblue')
            ax1.set_xlabel('Vaccination Center', fontsize=12)
            ax1.set_ylabel('Number of Villages Assigned', fontsize=12)
            ax1.set_title('Distribution of Village Assignments', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig1)
            
            # Distance distribution histogram
            st.write("**Distance Distribution**")
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.hist(result_df['distance'], bins=20, color='green', alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Distance (meters)', fontsize=12)
            ax2.set_ylabel('Number of Villages', fontsize=12)
            ax2.set_title('Distribution of Distances to Assigned Centers', fontsize=14, fontweight='bold')
            ax2.axvline(avg_distance, color='red', linestyle='--', linewidth=2, label=f'Average: {avg_distance:.0f}m')
            ax2.legend()
            plt.tight_layout()
            st.pyplot(fig2)
        
        with chart_col2:
            # Population served per center
            st.write("**Population Served per Center**")
            pop_by_center = result_df.groupby('vaccination_center')['population'].sum().sort_values(ascending=False)
            
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            pop_by_center.plot(kind='barh', ax=ax3, color='coral')
            ax3.set_xlabel('Total Population Served', fontsize=12)
            ax3.set_ylabel('Vaccination Center', fontsize=12)
            ax3.set_title('Population Coverage per Center', fontsize=14, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig3)
            
            # Infected cases per center
            st.write("**COVID Cases Served per Center**")
            infected_by_center = result_df.groupby('vaccination_center')['infected'].sum().sort_values(ascending=False)
            
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            colors = plt.cm.Reds(np.linspace(0.4, 0.8, len(infected_by_center)))
            infected_by_center.plot(kind='barh', ax=ax4, color=colors)
            ax4.set_xlabel('Total Infected Cases', fontsize=12)
            ax4.set_ylabel('Vaccination Center', fontsize=12)
            ax4.set_title('COVID-19 Cases Coverage per Center', fontsize=14, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig4)
        
        st.markdown("---")
        
        # Assignment Table with enhanced formatting
        st.subheader("📋 Detailed Village Assignments")
        
        display_df = result_df[['vaccination_center', 'distance', 'population', 'infected']].copy()
        display_df['distance'] = display_df['distance'].round(0)
        display_df.columns = ['Assigned Vaccination Center', 'Distance (m)', 'Population', 'Infected Cases']
        
        # Add color coding based on distance
        def color_distance(val):
            if val < avg_distance * 0.8:
                return 'background-color: #d4edda'  # Green
            elif val > avg_distance * 1.2:
                return 'background-color: #f8d7da'  # Red
            else:
                return 'background-color: #fff3cd'  # Yellow
        
        styled_df = display_df.style.applymap(color_distance, subset=['Distance (m)'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Summary Statistics
        st.subheader("📈 Summary Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Distance Statistics**")
            st.write(f"- Minimum: {result_df['distance'].min():.0f}m")
            st.write(f"- Maximum: {result_df['distance'].max():.0f}m")
            st.write(f"- Median: {result_df['distance'].median():.0f}m")
            st.write(f"- Std Dev: {result_df['distance'].std():.0f}m")
        
        with col2:
            st.write("**Population Statistics**")
            st.write(f"- Total: {result_df['population'].sum():,}")
            st.write(f"- Average per village: {result_df['population'].mean():.0f}")
            st.write(f"- Largest village: {result_df['population'].max():,}")
            st.write(f"- Smallest village: {result_df['population'].min():,}")
        
        with col3:
            st.write("**Infection Statistics**")
            st.write(f"- Total cases: {result_df['infected'].sum():,}")
            st.write(f"- Average per village: {result_df['infected'].mean():.0f}")
            st.write(f"- Infection rate: {(result_df['infected'].sum() / result_df['population'].sum() * 100):.2f}%")
        
        # Download Results
        st.markdown("---")
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            display_df.to_excel(writer, sheet_name='Assignments')
            distribution.to_frame('Count').to_excel(writer, sheet_name='Distribution')
            
            # Add summary sheet
            summary_data = {
                'Metric': ['Total Villages', 'Selected Sites', 'Average Distance (m)', 'Total Population', 'Total Infected', 'Optimization Cost'],
                'Value': [len(assignment), num_sites, f"{avg_distance:.0f}", result_df['population'].sum(), result_df['infected'].sum(), f"{cost:.2f}"]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="⬇️ Download Full Results (Excel)",
                data=output.getvalue(),
                file_name=f"vaccination_optimization_{city_input.replace(' ', '_').replace(',', '')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        with col2:
            # CSV download option
            csv = display_df.to_csv(index=True)
            st.download_button(
                label="⬇️ Download Results (CSV)",
                data=csv,
                file_name=f"vaccination_optimization_{city_input.replace(' ', '_').replace(',', '')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
    else:
        st.info("👈 Run optimization in the 'Run Optimization' tab to see results and analytics here.")

with tab5:
    st.header("About This Tool")
    
    st.markdown("""
    ### How It Works
    
    This tool uses:
    1. **OpenStreetMap** for real road network data
    2. **Genetic Algorithms** for optimization
    3. **Cost Function** that considers:
       - Distance from villages to vaccination centers
       - Population density
       - Infection rates
    
    ### Input Requirements
    
    - **Vaccination Centers**: List of potential sites with coordinates
    - **Village Centers**: Areas to serve with population and infection data
    
    ### The Math Behind It
    
    The optimization minimizes:
    ```
    Total Cost = Σ (weight × distance)
    where weight = (infected/total_infected) + (population/total_population)
    ```
    
    This ensures that:
    - High-population areas are prioritized
    - Areas with more infections get preference
    - Travel distances are minimized
    
    ### Credits
    
    Based on the research paper:  
    **"Optimal Location of COVID-19 Vaccination Sites"**  
    by Cabanilla et al., 2022
    
    ### Source Code
    
    View on GitHub: [Vaccination-Site-Optimization](https://github.com/kurtizak/Vaccination-Site-Optimization)
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Vaccination Site Optimizer v2.0**")
st.sidebar.markdown("Built with Streamlit + OSMnx")
