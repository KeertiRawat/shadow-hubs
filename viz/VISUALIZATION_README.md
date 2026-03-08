# Global Oil Trade Shadow Hubs 3D Visualization

## Overview
An interactive 3D globe visualization built with Globe.gl that displays global oil trade flows and identifies "shadow hub" countries in the network. This is a single-file, self-contained HTML visualization suitable for class projects analyzing oil trade networks.

## Features

### 1. 3D Rotating Globe
- Real-time rendered 3D Earth using WebGL
- Auto-rotating globe for dynamic visualization
- Zoom and pan controls

### 2. Trade Flow Visualization
- **Arcs**: Curved lines representing oil trade flows from exporter to importer countries
- **Arc Color**: Represents trade value magnitude
  - Red/bright red: Top 20% flows
  - Orange: 20-50% range
  - Blue: Lower value flows
- **Arc Width**: Proportional to trade volume (0.4-2.4 pixel range)
- **Animation**: Smooth dashed line animation for visual continuity
- **Filtering**: Shows top 200 flows per year to maintain performance

### 3. Country Node Points
- **Size**: Proportional to shadow hub residual score (bigger = more "shadow hub-like")
- **Color**: Based on OFAC entity presence
  - Red/bright red: High OFAC presence (>500 entities)
  - Orange: Medium OFAC (100-500 entities)
  - Blue/teal: Low or zero OFAC (<100 entities)

### 4. Shadow Hub Analysis
Countries are scored for "shadow hub" behavior using:
- **Shadow Residual Score**: Deviation from expected betweenness centrality
- **Shadow Rank**: Ranking of countries by residual (lower number = more shadow-like)
- **Betweenness Centrality**: Network positioning metric
- **Trade Volume**: Total trade value USD

### 5. Interactive Controls
- **Year Selector**: Dropdown to choose 2019-2024
- **Real-time Statistics**:
  - Active flows count
  - Total trade volume
  - Countries involved
  - Top trade flow
- **Hover Tooltips**: Detailed country information on mouse hover

### 6. Visual Legend
- Color meanings for OFAC presence
- Size metrics explanation
- Clearly identifies which nodes/arcs represent what data

### 7. Dark Theme
Professional dark theme with:
- Deep blue/black background
- Light blue accents
- High contrast for data visibility
- Glassmorphic UI panels with backdrop blur

## Data Structure

### Embedded Data
All data is embedded as JavaScript objects within the HTML:

1. **countriesData** (230 countries)
   - ISO3 code, name, coordinates (lat/lon)
   - OFAC entity counts
   
2. **edgesByYear** (2019-2024)
   - Top 200 trade flows per year
   - Exporter/importer ISO3 codes
   - Trade value in USD

3. **shadowLookup** (Shadow hub metrics by year)
   - Shadow residual score
   - Shadow rank
   - Betweenness centrality
   - Trade volume
   - Degree metrics

## Usage

### Opening the Visualization
1. Open `globe_viz.html` in a modern web browser
2. The globe will load and begin auto-rotating
3. Select a year from the dropdown (default: 2019)
4. Interact with the globe:
   - Click and drag to rotate
   - Scroll to zoom in/out
   - Hover over countries for details

### Interpreting the Data

**Shadow Hub Identification**
- Countries with high positive residual scores are "shadow hubs"
- They have higher betweenness centrality than expected given their trade volume
- This suggests they serve disproportionately as intermediaries in oil trade networks
- High OFAC entity presence + high shadow residual = potential sanctions circumvention hub

**Trade Patterns**
- Bright red arcs = major trade corridors
- Blue arcs = minor or secondary flows
- Flow density shows market concentration
- Year-to-year changes reveal geopolitical and market shifts

**Geographic Insights**
- Size of nodes indicates their role in the network
- Clustering of flows shows regional trade relationships
- Arc pathways reveal strategic routes and choke points

## Technical Details

### Technologies Used
- **Globe.gl**: 3D globe visualization library
- **Three.js**: WebGL rendering engine (bundled with Globe.gl)
- **Vanilla JavaScript**: No frameworks, pure client-side rendering
- **CSS3**: Modern styling with glassmorphic effects

### File Size
- ~240 KB single-file HTML
- All data embedded (no external API calls required)
- Works completely offline after initial load

### Browser Compatibility
- Chrome/Chromium 90+
- Firefox 88+
- Safari 14+
- Edge 90+
- Requires WebGL support

### Performance
- Renders 200 arcs per year smoothly
- 230+ countries with shadow metrics
- 60 FPS target on modern hardware
- Optimized data structure for fast lookup

## Data Sources

1. **edges_country_oil_2019plus.csv**
   - 45,513 trade flow records
   - Year, exporter ISO3, importer ISO3, trade value USD
   - Years: 2019-2024

2. **nodes_country_oil_2019plus.csv**
   - 232 countries/entities
   - OFAC entity counts and links
   - Group flag to filter regions vs. countries

3. **shadow_hubs_residual_2019plus.csv**
   - 413 shadow hub scoring records
   - Betweenness, residuals, rankings by year
   - Trade volume and degree metrics

## Customization Notes

### To modify the visualization:
1. Edit the HTML file in a text editor
2. Modify the embedded data in the `<script>` tag
3. Adjust colors in the JavaScript color functions
4. Change year range by editing the year selector options
5. Modify scaling factors for arc width, node size, etc.

### Common modifications:
- **Arc color scheme**: Edit `getArcColor()` function
- **Node sizing**: Edit `getPointSize()` function
- **OFAC thresholds**: Edit `getPointColor()` function
- **Animation speed**: Modify `arcDashAnimateTime`
- **Globe rotation speed**: Adjust the rotation interval calculation

## Class Project Notes

This visualization is ideal for demonstrating:
1. Network analysis concepts (betweenness, centrality)
2. Geopolitical risk (OFAC presence and trade patterns)
3. Data visualization techniques
4. Interactive JavaScript applications
5. Real-world applications of network science

The "shadow hub" concept shows how statistical residual analysis can identify anomalous network behavior, useful for detecting sanctions evasion or trade circumvention.

## License and Attribution

Created for academic analysis of global oil trade networks.
Uses Globe.gl library (MIT License): https://github.com/vasturiano/globe.gl

---
Generated: March 1, 2026
