# Quick Demo Script for Riyadh

## Step-by-Step Instructions

### 1. Select Riyadh from Dropdown
1. Open http://localhost:8501
2. In the sidebar, click the "Select City/Region" dropdown
3. Choose **"Riyadh, Saudi Arabia"**
4. You'll see "📍 Selected: Riyadh, Saudi Arabia"

### 2. Auto-Fetch Real Facilities
1. Go to **"Data Input"** tab
2. Under "💉 Vaccination Centers", click **"🔍 Auto-Extract Facilities from OpenStreetMap"**
3. Wait ~10 seconds while it fetches real hospitals and clinics from Riyadh
4. You'll see a list of actual hospitals with real coordinates!

### 3. Auto-Fetch Real Districts
1. Under "🏘️ Village/District Centers", click **"🏙️ Auto-Fetch Districts/Neighborhoods"**
2. The system will load 10 major Riyadh districts with:
   - Real coordinates (Al Olaya, Al Malaz, etc.)
   - Realistic population data
   - COVID infection estimates

### 4. Run the Optimization
1. Go to **"Run Optimization"** tab
2. Set parameters in sidebar:
   - Number of sites: 2 or 3
   - Fast Mode: ✅ (checked)
   - Distance: road
3. Click **"🚀 Start Optimization"**
4. Watch the progress bar!
5. 🎈 Balloons when complete!

### 5. View Beautiful Results
1. Go to **"Results & Analytics"** tab
2. You'll see:
   - ✅ **Interactive Map** showing optimal centers and districts
   - ✅ **4 Statistical Charts**:
     - Distribution of village assignments (bar chart)
     - Distance distribution (histogram)
     - Population coverage (horizontal bar chart)
     - COVID cases per center (color-coded chart)
   - ✅ **Color-coded table** (green = close, red = far)
   - ✅ **Summary statistics** (distance, population, infections)
   - ✅ **Download buttons** for Excel and CSV

## What Makes This Real Data?

### Vaccination Centers (Hospitals)
- Pulled directly from OpenStreetMap database
- Real hospital names in Riyadh
- Actual GPS coordinates
- Examples: King Fahad Medical City, King Faisal Specialist Hospital

### Districts
- 10 major Riyadh neighborhoods:
  - Al Olaya (Financial District)
  - Al Malaz  
  - Al Muruj
  - Al Naseem
  - Al Suwaidi
  - Al Aziziyah
  - Al Rabwa
  - Al Hamra
  - King Fahd District
  - Diplomatic Quarter
- Real coordinates for each district
- Population based on actual demographic data
- COVID rates estimated at 1-2% (realistic)

## Charts Explained

### 1. Villages per Vaccination Center
- Shows how many districts each selected hospital serves
- Helps identify if workload is balanced

### 2. Distance Distribution
- Histogram showing how far people need to travel
- Red line = average distance
- Ideal: Most bars to the left (short distances)

### 3. Population Served per Center
- Shows which hospitals serve the most people
- Helps with capacity planning

### 4. COVID Cases Served per Center
- Color intensity = number of cases
- Helps prioritize high-infection areas

## Tips for Great Demo

✅ **Use Riyadh** - Pre-loaded with realistic data  
✅ **Show the auto-fetch** - Impresses audience  
✅ **Explain the colors** on the map (each color = 1 center)  
✅ **Point out the charts** - Professional, data-driven  
✅ **Download the Excel** - Show it's production-ready  

## Backup Cities

If Riyadh doesn't work:
1. Manila, Philippines ✅
2. Dubai, United Arab Emirates ✅
3. Default: San Juan (always works) ✅

## Common Questions

**Q: Is this real data?**  
A: Yes! Hospitals are from OpenStreetMap, districts are real Riyadh neighborhoods, population estimates are realistic.

**Q: Can I use my own data?**  
A: Yes! Upload Excel files with your coordinates, or type a custom city name.

**Q: How accurate is the optimization?**  
A: Uses real road networks and genetic algorithms. Near-optimal solution in ~30 seconds.

## Sample Narration

"Let me demonstrate with Riyadh, Saudi Arabia. 

First, I'll select Riyadh from the dropdown. Now, I'll auto-extract real hospitals from OpenStreetMap... Excellent! It found 15 hospitals in Riyadh.

Next, I'll auto-fetch the major districts... Perfect! Here are 10 key neighborhoods with population and COVID data.

Now I'll run the optimization to find the best 2 vaccination centers. Watch the progress bar... optimizing... Complete!

Look at the results: The interactive map shows which hospital serves which district. 

The charts show us:
- Distribution of assignments
- Distance histogram - most people travel less than 5km
- Population coverage - well balanced
- COVID cases prioritized in high-infection areas

And I can download all this to Excel for reporting."

🎯 **Result**: Professional, data-driven demonstration!
