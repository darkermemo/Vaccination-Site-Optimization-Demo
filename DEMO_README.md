# Vaccination Site Optimization - Interactive Web Demo

## Overview
This interactive web application helps optimize COVID-19 vaccination site placement using geographic optimization and genetic algorithms.

## Features
- 🗺️ **Interactive Map Interface**: Visualize vaccination centers and village locations
- 🔍 **Auto-Geocoding**: Automatically fetch locations from OpenStreetMap
- 📊 **Progress Tracking**: Real-time progress bars during optimization
- 🎯 **Multiple Use Cases**: Easily switch between different cities and regions
- 📈 **Visual Results**: Interactive maps showing optimal assignments

## How It Works

### Input Data Requirements

The system needs two types of data:

1. **Vaccination Centers** (Potential Sites)
   - Name of the facility
   - Latitude and Longitude coordinates
   
2. **Village Centers** (Areas to Serve)
   - Village/Area name
   - Population count
   - Number of COVID-19 infected cases
   - Latitude and Longitude coordinates

### Example Input Format

**Vaccination Centers:**
```
| Name                          | latitude | longitude |
|-------------------------------|----------|-----------|
| San Juan Rural Health Unit II | 13.7913  | 121.408   |
| San Juan Doctors Hospital     | 13.8265  | 121.41    |
```

**Village Centers:**
```
| Village_name  | population | infected | latitude | longitude |
|---------------|------------|----------|----------|-----------|
| Abung         | 2444       | 14       | 13.7685  | 121.416   |
| Balagbag      | 2929       | 7        | 13.8032  | 121.416   |
```

## Running the Demo

### Option 1: Streamlit Web App (Recommended for Presentations)

```bash
# Install dependencies
pip install streamlit folium streamlit-folium osmnx geneticalgorithm openpyxl

# Run the web app
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### Option 2: Command Line

```bash
python run_optimization.py
```

## How to Add a New City

### Method 1: Using the Web Interface
1. Open the Streamlit app
2. Enter your city name (e.g., "Manila, Philippines")
3. The app will auto-detect the region from OpenStreetMap
4. Add vaccination centers manually or upload an Excel file
5. Add village centers with population and infection data
6. Click "Optimize" to see results

### Method 2: Creating Excel Files
1. Create `Vaccination_Centers_Table.xlsx` with columns: Name, latitude, longitude
2. Create `Village_Centers_Table.xlsx` with columns: Village_name, population, infected, latitude, longitude
3. Get coordinates from Google Maps (right-click → "What's here?")
4. Run the optimization

### Method 3: Auto-Extract from OpenStreetMap
The app can automatically extract hospitals and schools from any city:
```python
from app import auto_extract_facilities

facilities = auto_extract_facilities("Manila, Philippines")
```

## Understanding the Results

### The Map Visualization
- **Stars (⭐)**: Optimal vaccination centers selected
- **Dots (●)**: Village centers
- **Colors**: Villages assigned to the same vaccination center share the same color
- **Road Network**: Gray lines show the actual road network used for distance calculations

### The Assignment Table
Shows which villages should go to which vaccination center:
```
| Village      | Vaccination Center           | Distance (m) |
|--------------|------------------------------|--------------|
| Abung        | San Juan Rural Health Unit I | 2,150        |
| Balagbag     | San Juan District Hospital   | 1,890        |
```

### Cost Function
The optimization minimizes:
```
Total Cost = Σ (weight × distance)
where weight = (infected/total_infected) + (population/total_population)
```

This balances:
- **Proximity**: Shorter travel distances
- **Need**: Areas with more infections get priority
- **Capacity**: Larger populations are considered

## Demo Use Cases

### 1. Urban Healthcare Planning
**Scenario**: Optimize vaccination centers in a metropolitan area
**Input**: 50 potential sites, 200 neighborhoods
**Output**: Best 10 sites to minimize travel distance

### 2. Rural Health Outreach
**Scenario**: Limited facilities serving dispersed villages
**Input**: 5 mobile clinics, 30 remote villages
**Output**: Optimal deployment of mobile units

### 3. Emergency Response
**Scenario**: Rapid setup during outbreak
**Input**: Available facilities, outbreak hotspots
**Output**: Priority deployment locations

## Troubleshooting

### Common Issues

**"No network found for this location"**
- Solution: Try a more specific location name (e.g., "Los Baños, Laguna, Philippines" instead of just "Los Baños")

**"Optimization taking too long"**
- Solution: Use `fast_run=True` mode for demos (5 iterations instead of 600)
- Reduce the number of sites to optimize (L parameter)

**"Connection timeout"**
- Solution: Download data once, save locally, work offline

## Parameters Explained

- **L**: Number of vaccination sites to select (e.g., L=2 means pick best 2 out of all available)
- **distance**: Type of distance metric
  - `road`: Actual road network distance (realistic)
  - `euclidean`: Straight-line distance (faster)
  - `time`: Travel time based on road speeds
- **fast_run**: Use for demos (True) or production (False)
- **enumerative**: Try all combinations (slow, guaranteed optimal) vs genetic algorithm (fast, near-optimal)

## File Structure

```
Vaccination-Site-Optimization/
├── app.py                          # Streamlit web interface
├── run_optimization.py             # Core optimization engine
├── Vaccination_Centers_Table.xlsx  # Sample vaccination sites
├── Village_Centers_Table.xlsx      # Sample village data
├── README.md                       # This file
└── cache/                          # Cached map data (auto-generated)
```

## Credits

Based on the research paper:
**"Optimal Location of COVID-19 Vaccination Sites"** by Cabanilla et al., 2022

## License

MIT License - Feel free to use for academic presentations and research.
