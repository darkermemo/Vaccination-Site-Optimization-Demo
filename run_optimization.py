import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import osmnx as ox
import networkx as nx
from datetime import timedelta
from shapely.geometry import Polygon, LineString, Point
import itertools as iter
from geneticalgorithm import geneticalgorithm as ga
import math

# Configure osmnx
ox.settings.log_console = True

def site_extraction(place = "San Juan, Batangas, Philippines", 
                    tags = { "amenity": ["hospital", "school", "kindergarten", "college", "university"]}):
  
  ## Automated extraction of schools and hospitals

    tags = tags
    place = place
    # get all building footprints in some neighborhood
    # df = ox.geometries_from_place(place, tags) # Deprecated
    df = ox.features_from_place(place, tags)
    df["latlong"] = df.geometry.centroid
    df.reset_index(level=0, inplace=True)
    #df =df[["name", "latlong"]]
    return df

def Lsite_optimization(vacc, vill, 
                       L = 3, graph_area = ("San Juan, Batangas, Philippines"),
                        distance = "road", return_distances = False, enumerative = False, fast_run=False):
  '''
  @vacc =   Vaccination centers dataframe
  @villages =  Village centers dataframe
  @L =  number of vaccination sites to be used for optimization
  graph_area = string of the region name
  distance = string of the distance type, can be "road", "euclidean", or 'time
  return_distances = If true, the function will return the distances matrix between each village center and each vaccination site
  enumerative = If true, every possible combination of vaccination centers will be iterated on. 
                 If false, genetic algorithms will be used for speed.  
  '''
  
  TI = vill.infected.sum()
  TP = vill.population.sum() 
  vill["weight"] = vill.infected/TI + vill.population/TP

  ##  Automated N-site Optimization!!! ================================================================

  ## Select Open Street Map code for the given province 

  graph_area = graph_area

  # Create the graph of the area from OSM data. It will download the data and create the graph
  G = ox.graph_from_place(graph_area, network_type='drive')
  G = ox.add_edge_speeds(G)
  G = ox.add_edge_travel_times(G)


  ## Phase 1: N-site  Distance Matrix------------------------------------------------------------------
  index = vacc.Name
  columns = vill.Village_name
  df_distances = pd.DataFrame(index=index, columns = columns)
  
  for i in vacc.index:
    for j in vill.index:
      if distance == "road":
        #Get pure road distance
        origin_node = ox.nearest_nodes(G, Y = vill.iloc[j].latitude, X = vill.iloc[j].longitude)
        destination_node = ox.nearest_nodes(G, Y = vacc.iloc[i].latitude, X = vacc.iloc[i].longitude)
        
        df_distances.iloc[i,j] = nx.shortest_path_length(G, 
                                            origin_node, destination_node, weight='length')
      if distance == "euclidean":
        try:
             df_distances.iloc[i,j] = ox.distance.euclidean(
                vacc.iloc[i].latitude, vacc.iloc[i].longitude, vill.iloc[j].latitude, vacc.iloc[j].longitude)
        except:
             # Manual euclidean
             df_distances.iloc[i,j] = ((vacc.iloc[i].latitude - vill.iloc[j].latitude)**2 + (vacc.iloc[i].longitude - vill.iloc[j].longitude)**2)**0.5

      if distance == "time":
        origin_node = ox.nearest_nodes(G, Y = vill.iloc[j].latitude, X = vill.iloc[j].longitude)
        destination_node = ox.nearest_nodes(G, Y = vacc.iloc[i].latitude, X = vacc.iloc[i].longitude)
        
        df_distances.iloc[i,j] = nx.shortest_path_length(G, 
                                            origin_node, destination_node, weight='travel_time')


  index1 = list(iter.combinations(df_distances.index, L))
  master = pd.DataFrame(index=index1, columns = vill.Village_name)

  print("phase 1 complete,  distance matrix computed")

  #Phase 2 ---------------------------------------------------------------------------

  if enumerative is True:
    for i, name in enumerate(list(iter.combinations(df_distances.index, L))):
        #print(i)
        for j in vill.index:
          # Get weights
          weight = vill["weight"][j]
          sads = [None]*L
          for k in np.arange(0,L):
            sads[k] = df_distances.loc[name[k]][j]
          master.iloc[i,j] = weight*min(sads)
    print("phase 2 complete, optimization finished")

    ## Display Best Vaccination centers! 
    results_Lsite = pd.DataFrame(master.sum(axis=1))
    results_Lsite.columns = ["Cost"]
    if return_distances is True:
      return results_Lsite.sort_values(by=['Cost'], ascending=True), df_distances
    else:
      return results_Lsite.sort_values(by=['Cost'], ascending=True) 


  else:

    ## Combinatorial Genetic Algorithm Optimization 
    varbound=np.array([[0,vacc.shape[0]-1]]*L)
    
    master = pd.DataFrame(0, index=[0], columns=vill.Village_name)
    def f(x):
      x = x.tolist()
      for j in vill.index:
        # Get weights
        weight = vill["weight"][j]
        sads = [None]*L
        for k in np.arange(0,L):     
          sads[k] = df_distances.iloc[math.ceil(x[k]),j]
        master.iloc[0,j] = weight*min(sads)
      fitness = master.sum(axis=1)[0]
      return fitness
    

    if fast_run:
        algorithm_param = {'max_num_iteration': 5,
                          'population_size': 10,
                          'mutation_probability':0.1,
                          'elit_ratio': 0.01,
                          'crossover_probability': 0.5,
                          'parents_portion': 0.3,
                          'crossover_type':'uniform',
                          'max_iteration_without_improv': 2}
    else:
        algorithm_param = {'max_num_iteration': 300*L,
                          'population_size': 20*(L**2),
                          'mutation_probability':0.1,
                          'elit_ratio': 0.01,
                          'crossover_probability': 0.5,
                          'parents_portion': 0.3,
                          'crossover_type':'uniform',
                          'max_iteration_without_improv': 50}

    
    model = ga(function= f, dimension= L, variable_type='int', variable_boundaries=varbound, 
            algorithm_parameters=algorithm_param)

    print(model.run())
    

    ## Display Best Vaccination centers! 
    results_Lsite =  [int(i) for i in model.output_dict['variable'].tolist()]
    results_Lsite = tuple(df_distances.index[results_Lsite].tolist())
    wew = {'Cost': model.output_dict['function']}
    results_Lsite = pd.DataFrame(wew, index = [results_Lsite])
    
    if return_distances is True:
      return results_Lsite, df_distances
    else:
      return results_Lsite

def site_distribution(results, df_distances):

  ## results = results from the nsite_optimization function
  ## df_distances = distance matrix from the nsite_optimization function
    solution = df_distances.loc[[a for a in results.index[0]]]
    index = df_distances.columns
    columns = ["vaccination_center", "distance"]
    assignment  = pd.DataFrame(index=index, columns = columns)
    for i, name in enumerate(df_distances.columns):
      assignment.loc[name, "vaccination_center"] = solution.loc[:, [name]].sort_values(by=[name], ascending= True).index[0]
      assignment.loc[name, "distance"] = solution.loc[:, [name]].sort_values(by=[name], ascending=True).iloc[0, 0]
    
    distribution = assignment["vaccination_center"].value_counts()
    return assignment, distribution


def optimal_sites(vaccination_centers_df, villages_df,
                       L = 3,  graph_area = ("San Juan, Batangas, Philippines"), return_ranking = False, enumerative = False,
                        distance = "road", return_distances = False, plot = True, fast_run=False):
  '''
  PARAMS: 
  @vaccinatiOn_centers_df =   Vaccination centers dataframe
  @villages_df =  Village centers dataframe
  @L =  number of vaccination sites to be used for optimization
  graph_area = string of the region name
  distance = string of the distance type, can be "road", "euclidean", or 'time
  return_distances = If true, the function will return the distances matrix between each village center and each vaccination site
  enumerative = If true, every possible combination of vaccination centers will be iterated on. 
                 If false, genetic algorithms will be used for speed.  
  return_ranking = If true, then the results from the enumeration will be the output. This can only be done when enumerative = True.
  plot = If true then geospatial plot is shown
  fast_run = If true, use fewer iterations for quick testing
  '''
  

  vacc = vaccination_centers_df
  vill = villages_df
  TI = vill.infected.sum()
  TP = vill.population.sum() 
  vill["weight"] = vill.infected/TI + vill.population/TP
  
  result_nsites, df_distances = Lsite_optimization(vacc = vacc, vill = vill, L = L, graph_area = graph_area,
                                                   distance = distance, return_distances = True, enumerative = enumerative, fast_run=fast_run)
  assignment_nsite, distribution_nsite = site_distribution(results =  result_nsites, df_distances = df_distances)


  ## Plotting!!! =================================================================================================

  #Create the graph of the area from OSM data. 
  G = ox.graph_from_place(graph_area, network_type='drive')
  G = ox.add_edge_speeds(G)
  G = ox.add_edge_travel_times(G)

  ## Get coordinates of optimal sites
  optimal_sites = list(assignment_nsite.vaccination_center.unique())
  optimal_sites_loc = vacc[vacc.Name.isin(optimal_sites)] 
  optimal_sites_loc["colors"] = optimal_sites_loc["Name"].astype('category').cat.codes
  optimal_sites_loc

  # Get coordinates and assignments of barangays 
  wew = pd.merge(assignment_nsite, vill, on = "Village_name")
  wew["colors"] = wew["vaccination_center"].astype('category').cat.codes
  wew = wew.rename(columns={'vaccination_center': 'assigned_vaccination_center'})

  if plot is True:
      plt.figure(figsize=(25, 17), dpi=300)
      fig, ax = ox.plot_graph(G, edge_color = None, show=False, close=False, bgcolor = "white",
                              edge_linewidth=1, node_size=1, figsize=(25, 20), dpi = 300)

      ax.scatter(wew.longitude, wew.latitude, s= 100, c=wew["colors"], linewidths=1, edgecolors = "black", cmap='viridis')
      ax.scatter(optimal_sites_loc.longitude, optimal_sites_loc.latitude, s = 800, edgecolors = "black",
                c=optimal_sites_loc["colors"], linewidths=2, marker = "*")
      # plt.show()
      plt.savefig('optimization_result.png')
      print("Plot saved to optimization_result.png")
      

  if return_distances is True:
    if return_ranking is True:
      return assignment_nsite, df_distances, result_nsites
    else:
      return assignment_nsite, df_distances
  else:
    if return_ranking is True:
     return assignment_nsite, result_nsites
    else:
      return assignment_nsite

if __name__ == "__main__":
    print("Loading data...")
    try:
        vacc = pd.read_excel('Vaccination_Centers_Table.xlsx')
        vill = pd.read_excel('Village_Centers_Table.xlsx')
        print("Data loaded successfully.")
        
        print("Running optimization (FAST MODE for DEMO)...")
        assignment = optimal_sites(L = 2, vaccination_centers_df = vacc, villages_df = vill,
                            graph_area = ("San Juan, Batangas, Philippines"), plot=True, fast_run=True)
        print("Optimization complete.")
        print(assignment)
    except Exception as e:
        print(f"An error occurred: {e}")
