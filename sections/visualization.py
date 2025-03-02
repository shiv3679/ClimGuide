import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# File path for CRU dataset
LOCAL_FILE = "datasets/india.1960.2022.tmp.nc"
DOWNLOAD_URL = "https://crudata.uea.ac.uk/cru/data/hrg/"  # Example link, update with actual dataset link

# ----------------------------------------
# 📌 Load NetCDF Dataset Function
# ----------------------------------------
def load_dataset():
    """Loads the CRU dataset and returns the NetCDF object with explanations."""

    st.markdown("## 📂 Loading the CRU Climate Dataset")
    st.markdown("""
    Before we can analyze climate trends, we need to load our dataset.  
    Here, we are using the **CRU dataset** that contains monthly temperature data for **India (1960-2022)**.
    
    **Why NetCDF format?**  
    - 📦 Stores multi-dimensional climate data efficiently  
    - 🔍 Supports **time-series, latitude, and longitude** data  
    - 🚀 Optimized for large-scale scientific analysis  

    We will load the dataset and extract the **temperature variable** (`tmp`) for analysis.  
    """)

    # Visual Cue for Data Loading
    st.info("⏳ **Loading Climate Data...** This may take a few seconds.")

    # Try loading the dataset
    try:
        data = xr.open_dataset(LOCAL_FILE)['tmp']
        st.success("✅ **Dataset loaded successfully!**")
        st.markdown("""
        - **Dataset Name:** `india.1960.2022.tmp.nc`  
        - **Contains:** Monthly mean temperatures  
        - **Time Period:** 1960 - 2022  
        - **Dimensions:** Time ⏳ | Latitude 📍 | Longitude 🌍  

        **Now that we have the dataset, let's explore it further!** 🔍
        """)

        return data

    except FileNotFoundError:
        st.error("⚠️ **NetCDF file not found!** Please ensure `india.1960.2022.tmp.nc` exists in the `datasets/` folder.")
        st.markdown("""
        📥 **Want to download the dataset?** You can find it on the  
        [CRU Data Repository](https://crudata.uea.ac.uk/cru/data/hrg/).  

        📌 **Once downloaded, place it in your `datasets/` folder and restart this page.**
        """)
        return None


# ----------------------------------------
# 🌎 Spatial Visualization Function
# ----------------------------------------
def show_spatial_visualization(tmp_data):
    """Generates a spatial map of mean temperature with explanations."""

    st.markdown("## 🌎 Spatial Visualization of Climate Data")
    st.markdown("""
    **Why Spatial Visualization?**  
    - Climate data varies **spatially** across regions.  
    - This helps us understand **temperature distributions** over an area.  
    - Scientists use **maps** to analyze trends, patterns, and climate anomalies.  

    In this section, we will create a **temperature map** using `Matplotlib` and `Cartopy`.
    """)

    st.info("✅ **We will now compute and visualize the mean temperature over time.**")

    # Compute Mean Temperature Over Time
    tmp_time_mean = tmp_data.mean(dim='time')

    # Extract latitude and longitude
    lat = tmp_data.lat
    lon = tmp_data.lon

    # 📌 **Explain the Code to Students**
    st.markdown("### 🔍 Understanding the Code")
    st.markdown("""
    - `tmp_data.mean(dim='time')` → Computes the **mean temperature** over all time steps.  
    - `fig, ax = plt.subplots(..., projection=ccrs.PlateCarree())` → Creates a **map projection** for plotting.  
    - `ax.imshow(...)` → Displays the temperature **spatially**, using latitude and longitude.  
    - `ax.add_feature(...)` → Adds **geographical features** (borders, coastlines, etc.).  
    - `fig.colorbar(...)` → Adds a **color scale** to interpret temperature values.  
    """)

    # Display Code Block for Users
    st.code("""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Compute mean over time
tmp_time_mean = tmp_data.mean(dim='time')

# Create a figure with a map projection
fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})

# Plot temperature data
mp = ax.imshow(tmp_time_mean, extent=(lon.min(), lon.max(), lat.min(), lat.max()), cmap='jet', origin='lower')

# Add features
ax.set_title('Mean Temperature (°C)')
ax.add_feature(cfeature.BORDERS, edgecolor='black')  # Borders between countries
ax.add_feature(cfeature.COASTLINE)  # Coastlines

# Add colorbar for temperature scale
cbar = fig.colorbar(mp, shrink=0.8, label='Temperature (°C)')
cbar.minorticks_on()

plt.show()
    """, language="python")

    st.success("📌 **This reusable code can be applied to any climate dataset!**")

    # 🖼 **Generate the Actual Visualization**
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    mp = ax.imshow(tmp_time_mean, extent=(lon.min(), lon.max(), lat.min(), lat.max()), cmap='jet', origin='lower')

    ax.set_title('Mean Temperature (°C)')
    ax.add_feature(cfeature.BORDERS, edgecolor='black')
    ax.add_feature(cfeature.COASTLINE)

    cbar = fig.colorbar(mp, shrink=0.8, label='Temperature (°C)')
    cbar.minorticks_on()

    st.pyplot(fig)

    # 📌 **Final Encouragement**
    st.markdown("""
    🌟 **Congratulations!** You have successfully visualized spatial climate data.  
    - **Try applying this method to other variables!**  
    - **Change the color maps (`cmap`) to visualize trends better.**  
    - **What happens when you compute mean over a different time range?**  

    🚀 **Experiment, explore, and analyze! The more you practice, the better you get.**
    """)



# ----------------------------------------
# 📈 Time-Series Analysis Function
# ----------------------------------------
def show_time_series_analysis(tmp_data):
    """Generates time-series plots for temperature trends with explanations."""

    st.markdown("## 📈 Time-Series Temperature Analysis (India)")
    st.markdown("""
    **Why Time-Series Analysis?**  
    - Climate change is best studied over **time**.  
    - Time-series plots help detect **trends, cycles, and anomalies**.  
    - We can apply **smoothing** to remove short-term fluctuations.  

    In this section, we will:
    - Compute **average temperature** over India.
    - Visualize **monthly trends**.
    - Apply **rolling means** to smooth fluctuations.
    """)

    st.info("✅ **Let's start by computing the average temperature over lat & lon.**")

    # Compute Mean Over Lat & Lon
    tmp_fldmean = tmp_data.mean(dim=['lat', 'lon'])

    # 📌 **Explain the Code to Students**
    st.markdown("### 🔍 Understanding the Code")
    st.markdown("""
    - `tmp_data.mean(dim=['lat', 'lon'])` → Computes the **average temperature** across India.  
    - `ax.plot(tmp_fldmean.time, tmp_fldmean, color='blue')` → Plots temperature **over time**.  
    - `plt.xlabel('Time')`, `plt.ylabel('Temperature')` → Labels **time & temperature axes**.  
    """)

    # Display Code Block for Users
    st.code("""
import matplotlib.pyplot as plt

# Compute mean temperature over lat & lon
tmp_fldmean = tmp_data.mean(dim=['lat', 'lon'])

# Plot time-series data
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(tmp_fldmean.time, tmp_fldmean, color='blue', label='Raw Data')
ax.set_title('Monthly Temperature Time-Series (India)')
ax.set_xlabel('Time')
ax.set_ylabel('Temperature (°C)')
ax.legend()

plt.show()
    """, language="python")

    # Generate the Actual Time-Series Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(tmp_fldmean.time, tmp_fldmean, color='blue', label='Raw Data')
    ax.set_title('Monthly Temperature Time-Series (India)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature (°C)')
    ax.legend()
    st.pyplot(fig)

    # ------------------------------------
    # 🔍 Smoothed Time-Series (Rolling Mean)
    # ------------------------------------
    st.markdown("## 🔍 Smoothed Time-Series (Rolling Mean)")
    st.markdown("""
    **Why Smooth the Data?**  
    - Monthly climate data has **short-term fluctuations**.  
    - Smoothing (rolling mean) removes noise to **highlight long-term trends**.  
    - We use a **12-month rolling mean** to capture **annual trends**.  
    """)

    st.info("✅ **Now, let's apply a rolling mean for better clarity.**")

    window_size = 12  # 1-year rolling mean
    tmp_fldmean_smoothed = tmp_fldmean.rolling(time=window_size, center=True).mean()

    # 📌 **Explain the Rolling Mean Code**
    st.markdown("### 🔍 Understanding the Rolling Mean Code")
    st.markdown("""
    - `.rolling(time=12).mean()` → Computes the **12-month moving average**.  
    - `color='lightgray'` → Original data in **light gray** for comparison.  
    - `color='blue'` → Smoothed data in **blue**.  
    """)

    st.code("""
# Apply rolling mean (smoothing)
window_size = 12  # 12-month rolling mean
tmp_fldmean_smoothed = tmp_fldmean.rolling(time=window_size, center=True).mean()

# Plot smoothed time-series
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(tmp_fldmean.time, tmp_fldmean, color='lightgray', alpha=0.5, label='Original Data')
ax.plot(tmp_fldmean.time, tmp_fldmean_smoothed, color='blue', label=f'{window_size}-Month Moving Average')
ax.set_title('Smoothed Temperature Time-Series')
ax.set_xlabel('Time')
ax.set_ylabel('Temperature (°C)')
ax.legend()

plt.show()
    """, language="python")

    # Generate Smoothed Time-Series Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(tmp_fldmean.time, tmp_fldmean, color='lightgray', alpha=0.5, label='Original Data')
    ax.plot(tmp_fldmean.time, tmp_fldmean_smoothed, color='blue', label=f'{window_size}-Month Moving Average')
    ax.set_title(f'Smoothed Temperature Time-Series (Rolling {window_size}-Month Mean)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature (°C)')
    ax.legend()
    st.pyplot(fig)

    # ------------------------------------
    # 📊 Annual Mean Temperature Trend
    # ------------------------------------
    st.markdown("## 📊 Annual Mean Temperature Trend")
    st.markdown("""
    Instead of monthly data, we can **aggregate temperatures yearly**.  
    - This helps detect **long-term warming trends**.  
    - We use `.resample(time='1Y').mean()` for yearly aggregation.  
    """)

    st.info("✅ **Let's now compute annual mean temperature trends.**")

    tmp_fldmean_annual = tmp_fldmean.resample(time='1Y').mean()

    # 📌 **Explain Resampling Code**
    st.markdown("### 🔍 Understanding Resampling Code")
    st.markdown("""
    - `.resample(time='1Y').mean()` → Groups data **by year** and computes mean.  
    - `ax.plot(..., color='red')` → Annual trend plotted in **red**.  
    """)

    st.code("""
# Resample data to annual means
tmp_fldmean_annual = tmp_fldmean.resample(time='1Y').mean()

# Plot annual trend
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(tmp_fldmean_annual.time, tmp_fldmean_annual, color='red', label='Annual Mean Temperature')
ax.set_title('Annual Mean Temperature Over Time')
ax.set_xlabel('Time')
ax.set_ylabel('Temperature (°C)')
ax.legend()

plt.show()
    """, language="python")

    # Generate Annual Mean Temperature Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(tmp_fldmean_annual.time, tmp_fldmean_annual, color='red', label='Annual Mean Temperature')
    ax.set_title('Annual Mean Temperature Over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature (°C)')
    ax.legend()
    st.pyplot(fig)

    # 📌 **Final Encouragement**
    st.markdown("""
    🎯 **What We Learned:**  
    - Time-series plots reveal **seasonal & long-term climate trends**.  
    - Rolling means help smooth fluctuations, revealing **clear patterns**.  
    - Resampling allows us to see **yearly temperature trends**.  

    🚀 **Want to explore further?**  
    - Try applying a **5-year rolling mean** instead of 12 months.  
    - Compare **different regions** (e.g., North vs. South India).  
    - Check how trends differ **before and after 2000**.  

    🔥 **Keep experimenting, and soon you'll be analyzing climate data like a pro!**  
    """)



# ----------------------------------------
# 🔍 Long-Term Temperature Trend Function
# ----------------------------------------
def show_temperature_trend(tmp_data):
    """Computes and displays a long-term temperature trend using linear regression."""
    
    st.markdown("## 🔍 Long-Term Temperature Trend Analysis")
    
    st.markdown("""
    **Why Analyze Long-Term Temperature Trends?**  
    - Detecting **climate change trends** is crucial for research & policy.  
    - Linear regression helps **quantify warming rates** over decades.  
    - Understanding trends allows us to **predict future climate patterns**.  
    """)

    st.info("✅ **Let's compute the long-term trend for India's temperature.**")

    # Step 1: Compute Annual Mean Temperature
    tmp_fldmean_annual = tmp_data.mean(dim=['lat', 'lon']).resample(time='1Y').mean()

    # Step 2: Convert Time to Numeric for Regression
    time_numeric = tmp_fldmean_annual.time.dt.year
    temperature_values = tmp_fldmean_annual.values

    # Step 3: Compute Linear Regression
    slope, intercept = np.polyfit(time_numeric, temperature_values, 1)
    trend_line = slope * time_numeric + intercept

    # 📌 **Explain the Code to Students**
    st.markdown("### 🔍 Understanding the Code")
    st.markdown("""
    - `tmp_data.mean(dim=['lat', 'lon'])` → Computes **mean temperature over India**.  
    - `.resample(time='1Y').mean()` → Aggregates data into **yearly averages**.  
    - `np.polyfit(time_numeric, temperature_values, 1)` → Computes **linear regression**.  
    - `ax.plot(..., color='red')` → Plots **temperature trend in red**.  
    - `ax.plot(..., color='black', linestyle='--')` → Plots **trend line in black dashed style**.  
    """)

    # Display Code Block for Users
    st.code("""
import numpy as np
import matplotlib.pyplot as plt

# Compute annual mean temperature
tmp_fldmean_annual = tmp_data.mean(dim=['lat', 'lon']).resample(time='1Y').mean()

# Convert time to numeric values (years)
time_numeric = tmp_fldmean_annual.time.dt.year
temperature_values = tmp_fldmean_annual.values

# Fit a linear regression
slope, intercept = np.polyfit(time_numeric, temperature_values, 1)

# Compute trend line
trend_line = slope * time_numeric + intercept

# Plot the trend
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(time_numeric, temperature_values, color='red', label='Annual Mean Temperature')
ax.plot(time_numeric, trend_line, color='black', linestyle='--', label=f'Trend ({slope:.4f}°C/year)')
ax.set_title('Annual Mean Temperature Over Time')
ax.set_xlabel('Year')
ax.set_ylabel('Temperature (°C)')
ax.legend()

plt.show()
    """, language="python")

    # 📊 **Generate the Visualization**
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_numeric, temperature_values, color='red', label='Annual Mean Temperature')
    ax.plot(time_numeric, trend_line, color='black', linestyle='--', label=f'Trend ({slope:.4f}°C/year)')
    ax.set_title('Annual Mean Temperature Over Time')
    ax.set_xlabel('Year')
    ax.set_ylabel('Temperature (°C)')
    plt.grid(True)
    ax.legend()
    st.pyplot(fig)

    # 📌 **Final Interpretation**
    st.markdown("### 📌 Interpretation of Results")
    st.markdown(f"""
    - The computed warming trend is **{slope:.4f}°C per year**.  
    - This suggests a total temperature increase of **{slope * (time_numeric[-1] - time_numeric[0]):.2f}°C** over the dataset period.  
    - A positive slope indicates **warming**, while a negative slope would indicate **cooling**.  
    """)

    # 🌍 **Final Encouragement**
    st.markdown("""
    🎯 **Key Takeaways:**  
    - Linear regression helps **quantify long-term warming**.  
    - Even **small temperature increases** have major climate impacts.  
    - With practice, you can **analyze trends globally & regionally**.  

    🔥 **Want to explore further?**  
    - Try analyzing **seasonal trends** instead of yearly.  
    - Compare **pre-2000 vs. post-2000 warming trends**.  
    - Check temperature trends for **different Indian states**.  

    🚀 **Keep exploring—climate data science is the key to understanding our planet!**  
    """)

# ----------------------------------------
# 🔥 Contour Plot for Temperature Data
# ----------------------------------------
def show_contour_plot(tmp_data):
    """Generates a contour plot of mean temperature over time."""
    
    st.markdown("## 🔥 Contour Plot: Mean Temperature Over Time")
    
    st.markdown("""
    **Why Use Contour Plots in Climate Science?**  
    - Contours help **visualize temperature gradients** over large areas.  
    - Useful for **identifying climate zones** (hot/cold regions).  
    - Helps detect **temperature anomalies & trends** in different locations.  
    """)

    st.info("✅ **Let's create a contour plot of India's mean temperature!**")

    # Step 1: Compute Mean Temperature Over Time
    temperature_mean = tmp_data.mean(dim='time')

    # Step 2: Extract Latitude and Longitude
    lat = tmp_data.lat
    lon = tmp_data.lon

    # 📌 **Explain the Code to Students**
    st.markdown("### 🔍 Understanding the Code")
    st.markdown("""
    - `temperature.mean(dim='time')` → Computes **mean temperature across all years**.  
    - `plt.contourf(...)` → Plots **filled contours** representing temperature zones.  
    - `ax.add_feature(...)` → Adds **coastlines & borders** for reference.  
    - `cbar.set_label('Temperature (°C)')` → Adds **color legend for interpretation**.  
    """)

    # Display Code Block for Users
    st.code("""
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Compute mean over time
temperature_mean = tmp_data.mean(dim='time')

# Create figure with Cartopy projection
fig, ax = plt.subplots(figsize=(10,6), subplot_kw={'projection': ccrs.PlateCarree()})

# Create filled contour plot
contour = ax.contourf(temperature_mean.lon, temperature_mean.lat, temperature_mean, 
                       levels=np.arange(temperature_mean.min(), temperature_mean.max(), 5), 
                       cmap='coolwarm', extend='both')

# Add coastlines and borders
ax.add_feature(cfeature.COASTLINE, linewidth=1)
ax.add_feature(cfeature.BORDERS, linewidth=0.5)

# Add colorbar
cbar = plt.colorbar(contour, orientation="vertical", pad=0.02)
cbar.set_label('Temperature (°C)')

# Set title and labels
ax.set_title('Mean Surface Temperature (1960-2022)')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Show plot
plt.show()
    """, language="python")

    # 📊 **Generate the Visualization**
    fig, ax = plt.subplots(figsize=(10,6), subplot_kw={'projection': ccrs.PlateCarree()})
    
    contour = ax.contourf(temperature_mean.lon, temperature_mean.lat, temperature_mean, 
                           levels=np.arange(temperature_mean.min(), temperature_mean.max(), 5), 
                           cmap='coolwarm', extend='both')

    ax.add_feature(cfeature.COASTLINE, linewidth=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)

    cbar = plt.colorbar(contour, orientation="vertical", pad=0.02)
    cbar.set_label('Temperature (°C)')

    ax.set_title('Mean Surface Temperature (1960-2022)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    st.pyplot(fig)

    # 📌 **Final Interpretation**
    st.markdown("### 📌 Interpretation of Contour Plots")
    st.markdown("""
    - Warmer colors (red) indicate **higher temperatures**, while cooler colors (blue) indicate **lower temperatures**.  
    - This allows us to **see heat distribution across India** over the long term.  
    - Contour plots **help scientists detect climate shifts over decades**.  
    """)

    # 🌍 **Final Encouragement**
    st.markdown("""
    🎯 **Key Takeaways:**  
    - Contour plots are **powerful tools** for climate data visualization.  
    - They reveal **temperature gradients and regional variations**.  
    - Try exploring **different climate datasets** to uncover new patterns! 🚀  
    """)


# ----------------------------------------
# 💨 Quiver Plot for Wind Vectors
# ----------------------------------------
def show_quiver_plot():
    """Generates a quiver plot to visualize wind vectors over India."""
    
    st.markdown("## 💨 Wind Vector Visualization (Quiver Plot)")

    st.markdown("""
    **Why Use Quiver Plots in Climate Science?**  
    - Wind vectors help **analyze atmospheric circulation**.  
    - Useful for studying **monsoons, cyclones, and wind-driven weather patterns**.  
    - Helps detect **wind speed variations and climate shifts**.  
    """)

    st.info("✅ **Let's create a quiver plot to visualize India's wind patterns!**")

    # Load the datasets
    try:
        uwnd_ds = xr.open_dataset('./datasets/india.uwnd.nc', decode_times=False)
        vwnd_ds = xr.open_dataset('./datasets/india.vwnd.nc', decode_times=False)
    except FileNotFoundError:
        st.error("⚠️ Wind data files not found! Ensure `india.uwnd.nc` and `india.vwnd.nc` exist in `datasets/`.")
        return

    # Extract U and V wind components and compute the mean over time
    uwnd = uwnd_ds['uwnd'].mean(dim='time').squeeze()
    vwnd = vwnd_ds['vwnd'].mean(dim='time').squeeze()

    # Convert 1D lat/lon to 2D grid
    lons, lats = np.meshgrid(uwnd.lon, uwnd.lat)

    # 📌 **Explain the Code to Students**
    st.markdown("### 🔍 Understanding the Code")
    st.markdown("""
    - `xr.open_dataset(...)` → Loads **U (east-west) and V (north-south) wind datasets**.  
    - `mean(dim='time')` → Computes **average wind vectors over time**.  
    - `plt.quiver(...)` → Plots **arrows representing wind direction & speed**.  
    """)

    # Display Code Block for Users
    st.code("""
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Load wind datasets
uwnd_ds = xr.open_dataset('./datasets/india.uwnd.nc', decode_times=False)
vwnd_ds = xr.open_dataset('./datasets/india.vwnd.nc', decode_times=False)

# Compute mean wind components over time
uwnd = uwnd_ds['uwnd'].mean(dim='time').squeeze()
vwnd = vwnd_ds['vwnd'].mean(dim='time').squeeze()

# Convert 1D lat/lon to 2D grid
lons, lats = np.meshgrid(uwnd.lon, uwnd.lat)

# Create figure with Cartopy projection
fig, ax = plt.subplots(figsize=(10,6), subplot_kw={'projection': ccrs.PlateCarree()})

# Add map features
ax.add_feature(cfeature.COASTLINE, linewidth=1)

# Create quiver plot (Wind Vectors)
plt.quiver(lons, lats, uwnd, vwnd, transform=ccrs.PlateCarree(), 
           scale=200, width=0.002, color='black')

# Set title and labels
ax.set_title('Mean Wind Vectors over India')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Show plot
plt.show()
    """, language="python")

    # 📊 **Generate the Visualization**
    fig, ax = plt.subplots(figsize=(10,6), subplot_kw={'projection': ccrs.PlateCarree()})
    
    ax.add_feature(cfeature.COASTLINE, linewidth=1)

    plt.quiver(lons, lats, uwnd, vwnd, transform=ccrs.PlateCarree(), 
               scale=200, width=0.002, color='black')

    ax.set_title('Mean Wind Vectors over India')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    st.pyplot(fig)

    # ------------------------------------
    # 🔥 Wind Speed & Wind Vector Overlay
    # ------------------------------------
    st.markdown("### 🔥 Wind Speed & Wind Vector Overlay")

    # Compute wind speed (Magnitude of U and V wind)
    wind_speed = np.sqrt(uwnd**2 + vwnd**2)

    st.code("""
# Compute wind speed (magnitude of U and V wind)
wind_speed = np.sqrt(uwnd**2 + vwnd**2)

# Create figure with Cartopy projection
fig, ax = plt.subplots(figsize=(10,6), subplot_kw={'projection': ccrs.PlateCarree()})

# Add map features
ax.add_feature(cfeature.COASTLINE, linewidth=1)
ax.add_feature(cfeature.BORDERS, linewidth=0.5)

# Create filled contour plot (Wind Speed)
contour = plt.contourf(lons, lats, wind_speed, levels=15, cmap='coolwarm', extend='both', transform=ccrs.PlateCarree())

# Add colorbar
cbar = plt.colorbar(contour, orientation="vertical", pad=0.02)
cbar.set_label('Wind Speed (m/s)')

# Overlay wind vectors (quiver)
plt.quiver(lons, lats, uwnd, vwnd, transform=ccrs.PlateCarree(), 
           scale=200, width=0.002, color='black', alpha=0.8)

# Set title and labels
ax.set_title('Wind Speed & Wind Vectors over India')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Show plot
plt.show()
    """, language="python")

    # 📊 **Generate the Wind Speed & Wind Vector Overlay**
    fig, ax = plt.subplots(figsize=(10,6), subplot_kw={'projection': ccrs.PlateCarree()})

    ax.add_feature(cfeature.COASTLINE, linewidth=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)

    contour = plt.contourf(lons, lats, wind_speed, levels=15, cmap='coolwarm', extend='both', transform=ccrs.PlateCarree())

    cbar = plt.colorbar(contour, orientation="vertical", pad=0.02)
    cbar.set_label('Wind Speed (m/s)')

    plt.quiver(lons, lats, uwnd, vwnd, transform=ccrs.PlateCarree(), 
               scale=200, width=0.002, color='black', alpha=0.8)

    ax.set_title('Wind Speed & Wind Vectors over India')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    st.pyplot(fig)

    # 📌 **Final Interpretation**
    st.markdown("### 📌 Interpretation of Wind Visualization")
    st.markdown("""
    - **Wind vectors (arrows) indicate direction & speed**.  
    - **Contours represent wind speed** – higher speeds in red, lower speeds in blue.  
    - Useful for **studying monsoon patterns, jet streams, and extreme wind events**.  
    """)

    # 🌍 **Final Encouragement**
    st.markdown("""
    🎯 **Key Takeaways:**  
    - Quiver plots are **essential for wind pattern analysis**.  
    - Helps understand **climate dynamics, storms, and global circulation**.  
    - Experiment with **different timeframes** to see seasonal wind variations! 🚀  
    """)






# ----------------------------------------
# 🚀 Final Streamlit Page Function
# ----------------------------------------
def show_visualization():
    """Main function to display the visualization page."""
    
    st.title("📊 Climate Data Visualization")
    tmp_data = load_dataset()
    if tmp_data is None:
        return
    
    show_spatial_visualization(tmp_data)
    show_time_series_analysis(tmp_data)
    show_temperature_trend(tmp_data)
    show_contour_plot(tmp_data)
    show_quiver_plot()

    tmp_data.close()
