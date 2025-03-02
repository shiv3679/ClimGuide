# 🌍 Climate Learning & Data Exploration Platform

## 📖 Introduction
Welcome to the **Climate Learning & Data Exploration Platform** – an **interactive educational tool** designed to help students and researchers **analyze, visualize, and explore climate datasets**. 

This platform provides an **engaging learning experience** using **Python, Xarray, Matplotlib, and Cartopy** for **climate science**. Whether you're a **beginner in climate data analysis** or an **advanced researcher**, this tool **simplifies working with NetCDF files** and helps you extract **meaningful insights from climate data**.

---

## 🚀 Features
- 📚 **Educational Modules** → Learn Python, NumPy, Xarray, Matplotlib, Cartopy, and SciPy for climate analysis.
- 📂 **NetCDF Data Exploration** → Upload & explore metadata of climate datasets.
- 📈 **Time-Series Visualization** → Plot long-term trends and seasonal variations.
- 🌎 **Spatial Mapping** → Generate interactive maps using Cartopy.
- 🎮 **Playground Mode** → Write, test, and run Python code interactively inside the app.
- 🛠️ **Advanced Analysis** → Perform custom slicing, resampling, regression analysis, and filtering.

---

## 🛠 Installation

### 🔹 **1. Clone the Repository**
```sh
git clone https://github.com/shiv3679/Unnamed.git
cd climate-learning-platform
```
### 🔹 2. Set Up a Virtual Environment (Recommended)

Using Conda:
```sh
conda create --name venv python=3.9
conda activate venv
```

### 🔹 3. Install Dependencies
Using pip:
```sh
pip install -r requirements.txt
```

### 🔹 4. Run the Streamlit App

```sh
streamlit run app.py
```

## 🎮 How to Use
### 📂 Upload NetCDF Files
1. Navigate to the **"Playground"** section.
2. Upload a **NetCDF climate dataset** (`.nc` format).
3. View **metadata, variables, dimensions, and attributes.**
### 🌎 Visualizing Climate Data
1. Choose Time-Series Analysis 📈 to analyze long-term temperature trends.
2. Select Spatial Visualization 🗺️ to generate interactive maps.
3. Try Contour Plots & Wind Vector Analysis to understand climate patterns.
### 📝 Python Code Playground
1. Write & execute custom Python scripts inside the app.
2. Experiment with Xarray, Matplotlib, and SciPy interactively.

## 🌟 Sample Code for Climate Data Visualization

```python
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Load NetCDF dataset
data = xr.open_dataset('datasets/india.1960.2022.tmp.nc')

# Extract temperature variable and compute mean over time
temperature = data['tmp'].mean(dim='time')

# Create map projection
fig, ax = plt.subplots(figsize=(8,6), subplot_kw={'projection': ccrs.PlateCarree()})

# Plot temperature data
mp = ax.imshow(temperature, cmap='coolwarm', origin='lower',
               extent=[data.lon.min(), data.lon.max(), data.lat.min(), data.lat.max()])

# Add coastlines and borders
ax.add_feature(cfeature.BORDERS, edgecolor='black')
ax.add_feature(cfeature.COASTLINE)

# Add colorbar
plt.colorbar(mp, label='Temperature (°C)')
plt.title('Mean Surface Temperature (1960-2022)')
plt.show()
```

## ⚡ Future Roadmap
- ✔️ Add more datasets for climate analysis.
- ✔️ Enable interactive 3D visualizations using Plotly.
- ✔️ Improve user interface & performance.
- ✔️ Provide more hands-on tutorials for climate data analysis.

## 📜 License
This project is open-source and licensed under the MIT License.

## 📞 Contact & Support

Maintainer: Shiv Shankar Singh

📧 Email: shivshankarsingh.py@gmail.com