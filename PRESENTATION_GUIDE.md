# Quick Start Guide for Your Presentation

## Demo Commands (Copy-Paste Ready)

### 1. Start the Interactive Web App
```bash
cd "/Users/yasser2/Documents/EMBA/Stat/Vaccination-Site-Optimization"
streamlit run app.py
```

### 2. Alternative: Run Command-Line Version
```bash
python3 run_optimization.py
```

## Pre-Presentation Checklist

- [ ] Test the web app once before presenting
- [ ] Have sample data files ready (already present)
- [ ] Prepare a backup city/region name (e.g., "Manila, Philippines")
- [ ] Close unnecessary applications to ensure smooth performance

## Demo Flow for Presentation

### Part 1: Show the Problem (2 minutes)
1. Open the app
2. Go to "About" tab
3. Explain the cost function and optimization goal

### Part 2: Show the Current Data (1 minute)
1. Go to "Data Input" tab
2. Show the sample vaccination centers
3. Show the sample village data
4. Explain the columns (population, infected, coordinates)

### Part 3: Run the Optimization (2 minutes)
1. Go to "Run Optimization" tab
2. Adjust settings in sidebar (L=2, Fast Mode ON)
3. Click "Start Optimization"
4. Watch the progress bar
5. Celebrate when balloons appear!

### Part 4: Show Results (3 minutes)
1. Go to "Results" tab
2. Show the interactive map:
   - Stars = selected vaccination centers
   - Circles = villages (colored by assignment)
   - Lines = connections
3. Show the metrics at the top
4. Download the Excel results

### Part 5: Demo with New City (3 minutes)
1. Sidebar: Change city to "Manila, Philippines" (or your choice)
2. Data Input tab: Click "Auto-Extract Facilities"
3. Wait for facilities to load
4. Manually add 2-3 village centers with fake data:
   - Village_name, population, infected, latitude, longitude
5. Run optimization again
6. Show how it works for ANY city

## Troubleshooting During Presentation

**If the app crashes:**
- Restart: `streamlit run app.py`
- Use backup: Show the existing PNG files (output.png, wow.png)

**If optimization takes too long:**
- Make sure Fast Mode is checked
- Reduce L to 2
- Use smaller datasets

**If auto-extract doesn't work:**
- Have backup: Use the existing Excel files
- Explain: "In practice, you can manually enter or import from Google Maps"

## Key Talking Points

✅ "This works for ANY city worldwide"  
✅ "Uses real road networks from OpenStreetMap"  
✅ "Considers both population and infection rates"  
✅ "Fast optimization using genetic algorithms"  
✅ "Interactive visualization for decision-makers"  
✅ "Export results to Excel for reports"

## After the Demo

Share the GitHub link:
https://github.com/kurtizak/Vaccination-Site-Optimization

## Emergency Backup

If live demo fails, show pre-generated images:
- `output.png` - Results map
- `wow.png` - Visualization
- `2sites_white.png` - Alternative view

Good luck! 🍀
