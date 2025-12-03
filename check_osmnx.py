import osmnx as ox
print(dir(ox))
try:
    print(dir(ox.distance))
except:
    print("ox.distance not found")
