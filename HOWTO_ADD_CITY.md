# Example: Adding a New City - Step by Step

## Method 1: Using the Web Interface (Easiest)

### For presentation, let's add Manila, Philippines

1. **Open the app** (already running at http://localhost:8501)

2. **In the sidebar**, change "City/Region Name" to:
   ```
   Manila, Philippines
   ```

3. **Go to "Data Input" tab**

4. **Click "Auto-Extract Facilities from OpenStreetMap"**
   - This will automatically find hospitals and clinics in Manila
   - You'll see a list of facilities with coordinates

5. **Add sample village data** (for demo purposes):
   Create a new Excel file `Manila_Villages.xlsx` with:
   
   | Village_name | population | infected | latitude  | longitude  |
   |--------------|------------|----------|-----------|------------|
   | Ermita       | 8,000      | 120      | 14.5833   | 120.9847   |
   | Malate       | 77,513     | 230      | 14.5719   | 120.9897   |
   | Paco         | 70,978     | 180      | 14.5833   | 120.9897   |
   | San Miguel   | 12,000     | 150      | 14.5914   | 120.9944   |
   | Sampaloc     | 192,843    | 400      | 14.6031   | 120.9914   |

6. **Upload this file** in the Village Centers section

7. **Go to "Run Optimization" tab**

8. **Click "Start Optimization"**

## Method 2: Getting Coordinates from Google Maps

1. Open Google Maps: https://maps.google.com
2. Search for your location
3. Right-click on the map → "What's here?"
4. Copy the coordinates (e.g., 14.5833, 120.9847)
5. Add to your Excel file

## Method 3: Create Excel Files Manually

### Vaccination_Centers_Table.xlsx format:
```
| Name                    | latitude | longitude |
|-------------------------|----------|-----------|
| Manila General Hospital | 14.5833  | 120.9847  |
| Jose Reyes Hospital     | 14.6031  | 120.9914  |
```

### Village_Centers_Table.xlsx format:
```
| Village_name | population | infected | latitude | longitude |
|--------------|------------|----------|----------|-----------|
| Ermita       | 8000       | 120      | 14.5833  | 120.9847  |
```

## Common Cities You Can Demo

### Philippines:
- Manila, Philippines
- Quezon City, Philippines
- Cebu City, Philippines
- Davao City, Philippines

### International:
- Singapore
- Bangkok, Thailand
- Jakarta, Indonesia
- Dubai, United Arab Emirates

## Quick Fake Data Generator

For demo purposes, you can use these realistic-looking numbers:

**Population**: Between 5,000 - 50,000
**Infected**: Between 50 - 500 (roughly 1-2% of population)

## Tips for Smooth Demo

1. **Pre-load data before presenting**
   - Test with your chosen city beforehand
   - Save the Excel files in advance

2. **Use Fast Mode**
   - Always check "Fast Mode" in the sidebar
   - This makes optimization complete in ~30 seconds instead of 5+ minutes

3. **Start with L=2**
   - Shows results quickly
   - Easy to visualize
   - Can increase to L=3 or L=4 if time permits

4. **Have backup data**
   - Keep the default San Juan data as fallback
   - Works 100% of the time since it's tested

## Example Script for Your Presentation

"Let me show you how this works for ANY city worldwide. I'll use Manila as an example.

[Change city name to Manila]

Now, let's automatically extract available hospitals...

[Click Auto-Extract]

Great! The system found 15 hospitals in Manila using OpenStreetMap. Now I'll upload village data with population and infection statistics.

[Upload Excel]

Perfect! Now let's run the optimization to find the best 2 vaccination centers out of these 15 options.

[Click Start Optimization]

Watch the progress bar - it's computing distances along real road networks... analyzing 15 potential sites... running the genetic algorithm...

[Wait for completion]

Excellent! Here are the results. The map shows:
- Red/Blue stars: The 2 optimal vaccination centers
- Colored circles: Villages assigned to each center
- The algorithm minimized both travel distance and prioritized high-infection areas

[Show metrics and download results]

This same process works for any city - Cairo, London, Tokyo - anywhere in the world."
