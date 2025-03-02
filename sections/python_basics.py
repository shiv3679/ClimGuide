

import streamlit as st

# 🎨 Apply Custom Styling for a Modern UI
st.markdown("""
    <style>
        .subchapter-box {
            background-color: #1e293b;
            padding: 10px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(255, 255, 255, 0.1);
            margin-bottom: 10px;
        }
        .fade-in {
            animation: fadeIn 0.5s ease-in-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
    </style>
""", unsafe_allow_html=True)

def show_python_basics():
    st.title("🐍 Python Basics for Climate Data Science")
    st.write("Python is a powerful programming language used for climate data analysis and visualization.")

    # -------------------------------------------------
    # 🔹 Overview of Key Libraries (Always Visible Above Dropdown)
    # -------------------------------------------------
    st.markdown('<div class="subchapter-box">', unsafe_allow_html=True)
    st.subheader("🔹 Overview of Key Libraries")
    st.markdown("""
    These libraries are essential for climate data analysis:
    - 📦 **Xarray** - Works with NetCDF climate datasets.
    - 🔢 **NumPy** - Provides numerical computations.
    - 📊 **Matplotlib** - Visualizes climate trends.
    - 🌎 **Cartopy** - Creates geospatial maps.
    - 📑 **Pandas** - Manages structured climate data.
    - 📈 **SciPy** - Supports scientific computing.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.code("""
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import scipy.stats as stats
    """, language="python")

    # Accordion-style Subchapter Selection
    with st.expander("📖 Select a Subchapter"):
        subchapters = {
            "📦 Xarray - Handling Multi-dimensional Data": show_xarray,
            "🔢 NumPy - Numerical Computing": show_numpy,
            "📊 Matplotlib - Climate Data Visualization": show_matplotlib,
            "🌎 Cartopy - Geospatial Mapping": show_cartopy,
            "📑 Pandas - Time-Series Data Management": show_pandas,
            "📈 SciPy - Scientific Computing": show_scipy
        }

        selected_subchapter = st.radio("", list(subchapters.keys()))

    # Smooth transition effect
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    subchapters[selected_subchapter]()  # Dynamically load selected subchapter
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# 📦 Xarray - Handling Multi-dimensional Data
# -------------------------------------------------
def show_xarray():
    st.subheader("📦 Xarray - Handling Multi-dimensional Data")
    
    st.markdown("""
    **Why Use Xarray?**
    - Xarray is designed to efficiently handle **multi-dimensional arrays** (such as NetCDF climate datasets).
    - It extends **Pandas-like functionality** to multi-dimensional data.
    - It allows easy **group-by, resampling, interpolation**, and other operations on labeled climate data.
    - Xarray supports **Dask**, enabling parallel computing and handling large datasets seamlessly.
    """)

    # 🔧 Installation Instructions
    st.markdown("### 🔧 Installation Methods")
    st.info("""
    **Preferred Installation:** Using Conda (Installs Dependencies Automatically)
    """)
    st.code("conda install -c conda-forge xarray", language="bash")

    st.info("""
    **Alternative Installation:** Using pip (Manual Dependency Management)
    """)
    st.code("pip install xarray", language="bash")

    st.markdown("""
    **Why use Conda?**
    - Conda installs dependencies like **NumPy, Pandas, and Dask** automatically.
    - Ensures better compatibility, especially when working with climate datasets.
    - Pip requires additional dependency management.
    """)

    # 📂 Loading NetCDF Data with Xarray
    st.markdown("### 📂 Loading NetCDF Data with Xarray")
    st.markdown("""
    **Xarray makes it easy to read, explore, and manipulate NetCDF climate datasets.**
    """)

    st.code("""
import xarray as xr

# Load a NetCDF file
file = "climate_data.nc"
data = xr.open_dataset(file)

# Display dataset information
print(data)
    """, language="python")

    # 📊 Viewing Dataset Structure
    st.markdown("### 📊 Understanding Xarray Dataset Structure")
    st.code("""
import xarray as xr

data = xr.open_dataset("climate_data.nc")

# Print dataset summary
print(data)

# View dataset variables
print(data.variables)

# Select a specific variable
temperature = data["temperature"]
print(temperature)
    """, language="python")

    # 🔍 Indexing & Selecting Data
    st.markdown("### 🔍 Indexing & Selecting Data in Xarray")
    st.markdown("""
    Xarray allows indexing similar to Pandas but in multiple dimensions.
    """)

    st.code("""
# Select a specific time slice
time_slice = data.sel(time="2023-01-01")

# Select data using coordinate-based indexing
spatial_data = data.sel(latitude=40.0, longitude=-100.0, method="nearest")

print(time_slice)
print(spatial_data)
    """, language="python")

    # 📈 Plotting with Xarray
    st.markdown("### 📈 Visualizing Climate Data with Xarray")
    st.markdown("""
    Xarray integrates seamlessly with **Matplotlib** for visualization.
    """)

    st.code("""
import matplotlib.pyplot as plt

# Select a temperature variable and plot it
data["temperature"].isel(time=0).plot()
plt.title("Temperature Distribution on First Time Step")
plt.show()
    """, language="python")

    # 🚀 Advanced: Grouping & Aggregating Data
    st.markdown("### 🚀 Advanced: Grouping & Aggregating Climate Data")
    st.markdown("""
    - Xarray provides **groupby** operations for climate data analysis.
    - Useful for calculating **monthly, seasonal, or yearly averages**.
    """)

    st.code("""
# Compute the monthly mean of temperature
monthly_avg = data["temperature"].groupby("time.month").mean()
print(monthly_avg)
    """, language="python")

    st.success("✅ **Xarray makes working with NetCDF climate datasets simple and efficient!**")

# -------------------------------------------------
# 🔢 NumPy - Numerical Computing
# -------------------------------------------------
def show_numpy():
    st.subheader("🔢 NumPy - Numerical Computing for Climate Data")
    
    st.markdown("""
    **Why Use NumPy?**
    - NumPy is the **foundation** for numerical computing in Python.
    - Provides fast and efficient **array operations** optimized for scientific computing.
    - Essential for **mathematical calculations, linear algebra, and statistics**.
    - Used in climate data analysis for **handling large numerical datasets efficiently**.
    """)

    # 🔧 Installation Instructions
    st.markdown("### 🔧 Installation Methods")
    st.info("""
    **Preferred Installation:** Using Conda (Installs Dependencies Automatically)
    """)
    st.code("conda install -c conda-forge numpy", language="bash")

    st.info("""
    **Alternative Installation:** Using pip (Manual Dependency Management)
    """)
    st.code("pip install numpy", language="bash")

    st.markdown("""
    **Why use Conda?**
    - Conda automatically handles dependencies and ensures compatibility.
    - Pip requires manual dependency management.
    """)

    # 📂 Creating and Working with NumPy Arrays
    st.markdown("### 📂 Creating and Manipulating NumPy Arrays")
    st.markdown("""
    NumPy arrays (`ndarray`) are **faster and more memory-efficient** than Python lists.
    """)

    st.code("""
import numpy as np

# Creating a NumPy array
temperature = np.array([15, 18, 20, 22, 19])
print("Temperature Array:", temperature)

# Checking array properties
print("Shape:", temperature.shape)
print("Data Type:", temperature.dtype)
    """, language="python")

    # 🔍 Basic NumPy Operations
    st.markdown("### 🔍 Basic Mathematical Operations in NumPy")
    st.markdown("""
    NumPy allows **efficient mathematical computations** on large datasets.
    """)

    st.code("""
# Compute the mean temperature
mean_temp = np.mean(temperature)
print(f"Mean Temperature: {mean_temp}")

# Compute standard deviation
std_temp = np.std(temperature)
print(f"Temperature Standard Deviation: {std_temp}")

# Element-wise operations
temperature_fahrenheit = (temperature * 9/5) + 32
print("Temperatures in Fahrenheit:", temperature_fahrenheit)
    """, language="python")

    # 📈 NumPy in Climate Science: Handling Large Data
    st.markdown("### 📈 NumPy for Climate Data Analysis")
    st.markdown("""
    NumPy is widely used in **climate science** for handling **large-scale numerical datasets**.
    """)

    st.code("""
# Simulating daily temperature data for a year (365 days)
daily_temps = np.random.normal(loc=15, scale=5, size=365)

# Compute statistics
mean_annual_temp = np.mean(daily_temps)
print(f"Mean Annual Temperature: {mean_annual_temp:.2f}°C")
    """, language="python")

    # 🏎️ Speed Comparison: NumPy vs Python Lists
    st.markdown("### 🏎️ Speed Comparison: NumPy vs Python Lists")
    st.markdown("""
    NumPy is significantly **faster than Python lists** due to optimized operations.
    """)

    st.code("""
import time

# Python List Calculation
py_list = list(range(1000000))
start = time.time()
py_list = [x * 2 for x in py_list]
end = time.time()
print(f"Python List Time: {end - start:.5f} seconds")

# NumPy Array Calculation
np_array = np.arange(1000000)
start = time.time()
np_array = np_array * 2
end = time.time()
print(f"NumPy Array Time: {end - start:.5f} seconds")
    """, language="python")

    # 🚀 Advanced NumPy: Linear Algebra & Matrix Operations
    st.markdown("### 🚀 Advanced NumPy: Linear Algebra & Matrix Operations")
    st.markdown("""
    - NumPy provides powerful **linear algebra** functions for climate modeling.
    - Used for **temperature trend predictions, simulations, and climate models**.
    """)

    st.code("""
# Creating a matrix (temperature variations over 3 days at 3 locations)
temperature_matrix = np.array([[15, 18, 20], [22, 19, 17], [25, 23, 21]])

# Compute matrix transpose
transpose = np.transpose(temperature_matrix)
print("Transpose:\n", transpose)

# Compute dot product
result = np.dot(temperature_matrix, transpose)
print("Dot Product:\n", result)
    """, language="python")

    st.success("✅ **NumPy makes numerical computations in climate science efficient and fast!**")

# -------------------------------------------------
# 📊 Matplotlib - Climate Data Visualization
# -------------------------------------------------
def show_matplotlib():
    st.subheader("📊 Matplotlib - Climate Data Visualization")
    
    st.markdown("""
    **Why Use Matplotlib?**
    - Matplotlib is the **most widely used** visualization library in Python.
    - Provides highly customizable **line plots, scatter plots, histograms, and bar charts**.
    - Essential for **visualizing climate trends and data patterns**.
    - Works seamlessly with **NumPy and Xarray** for plotting scientific data.
    """)

    # 🔧 Installation Instructions
    st.markdown("### 🔧 Installation Methods")
    st.info("""
    **Preferred Installation:** Using Conda (Ensures Compatibility)
    """)
    st.code("conda install -c conda-forge matplotlib", language="bash")

    st.info("""
    **Alternative Installation:** Using pip (Manual Dependency Management)
    """)
    st.code("pip install matplotlib", language="bash")

    st.markdown("""
    **Why use Conda?**
    - Conda ensures compatibility with **SciPy, NumPy, and other scientific libraries**.
    - Pip installation may require additional dependencies.
    """)

    # 📈 Creating a Simple Line Plot
    st.markdown("### 📈 Creating a Simple Line Plot")
    st.markdown("""
    Let's visualize **temperature variation over time** using a line plot.
    """)

    st.code("""
import numpy as np
import matplotlib.pyplot as plt

# Sample time series data
time = np.arange(0, 10, 1)
temperature = [15, 18, 21, 19, 17, 20, 22, 21, 19, 18]

# Plot the data
plt.figure(figsize=(8,5))
plt.plot(time, temperature, marker='o', linestyle='-', color='b', label="Temperature")
plt.xlabel('Time (Days)')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Variation Over Time')
plt.legend()
plt.grid()
plt.show()
    """, language="python")

    # 🔍 Scatter Plots for Climate Data
    st.markdown("### 🔍 Scatter Plots for Climate Data")
    st.markdown("""
    Scatter plots help in analyzing **temperature trends and anomalies**.
    """)

    st.code("""
import matplotlib.pyplot as plt

# Random temperature data
time = np.arange(0, 20, 1)
temperature = np.random.normal(20, 5, size=len(time))

plt.figure(figsize=(8,5))
plt.scatter(time, temperature, color='r', marker='x', label="Temperature Readings")
plt.xlabel('Time (Days)')
plt.ylabel('Temperature (°C)')
plt.title('Scatter Plot of Temperature Readings')
plt.legend()
plt.grid()
plt.show()
    """, language="python")

    # 📊 Histograms: Temperature Distribution
    st.markdown("### 📊 Histograms: Temperature Distribution")
    st.markdown("""
    Histograms are useful for understanding **temperature distribution over a period**.
    """)

    st.code("""
import matplotlib.pyplot as plt

# Generate random temperature data
temperature_data = np.random.normal(20, 5, size=500)

plt.figure(figsize=(8,5))
plt.hist(temperature_data, bins=20, color='g', edgecolor='black', alpha=0.7)
plt.xlabel('Temperature (°C)')
plt.ylabel('Frequency')
plt.title('Histogram of Temperature Distribution')
plt.grid()
plt.show()
    """, language="python")

    # 📍 Bar Charts: Monthly Average Temperatures
    st.markdown("### 📍 Bar Charts: Monthly Average Temperatures")
    st.markdown("""
    Bar charts are great for **comparing monthly average temperatures**.
    """)

    st.code("""
import matplotlib.pyplot as plt

# Sample monthly temperature data
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
avg_temperatures = [3, 5, 9, 15, 20, 25, 28, 27, 22, 16, 9, 4]

plt.figure(figsize=(8,5))
plt.bar(months, avg_temperatures, color='c', alpha=0.8)
plt.xlabel('Month')
plt.ylabel('Average Temperature (°C)')
plt.title('Monthly Average Temperatures')
plt.grid(axis='y')
plt.show()
    """, language="python")

    # 🚀 Advanced: Subplots for Multi-variable Climate Data
    st.markdown("### 🚀 Advanced: Subplots for Multi-variable Climate Data")
    st.markdown("""
    Subplots allow us to visualize **multiple climate variables simultaneously**.
    """)

    st.code("""
import matplotlib.pyplot as plt

# Sample temperature and humidity data
time = np.arange(0, 10, 1)
temperature = np.random.normal(20, 5, size=len(time))
humidity = np.random.normal(60, 10, size=len(time))

fig, ax1 = plt.subplots(figsize=(8,5))

# Plot Temperature
ax1.set_xlabel('Time (Days)')
ax1.set_ylabel('Temperature (°C)', color='r')
ax1.plot(time, temperature, color='r', marker='o', linestyle='-', label="Temperature")
ax1.tick_params(axis='y', labelcolor='r')

# Create second y-axis for humidity
ax2 = ax1.twinx()
ax2.set_ylabel('Humidity (%)', color='b')
ax2.plot(time, humidity, color='b', marker='s', linestyle='--', label="Humidity")
ax2.tick_params(axis='y', labelcolor='b')

plt.title('Temperature and Humidity Over Time')
plt.grid()
plt.show()
    """, language="python")

    st.success("✅ **Matplotlib makes climate data visualization simple and effective!**")


# -------------------------------------------------
# 📑 Pandas - Time-Series Climate Data Management
# -------------------------------------------------
def show_pandas():
    st.subheader("📑 Pandas - Time-Series Climate Data Management")
    
    st.markdown("""
    **Why Use Pandas?**
    - Pandas is the **most powerful library** for handling structured datasets (CSV, Excel, JSON).
    - Provides **easy-to-use** data filtering, cleaning, and statistical analysis tools.
    - Works seamlessly with **NumPy and Matplotlib** for efficient climate data analysis.
    - Handles **time-series data**, which is crucial for climate science.
    """)

    # 🔧 Installation Instructions
    st.markdown("### 🔧 Installation Methods")
    st.info("""
    **Preferred Installation:** Using Conda (Ensures Compatibility)
    """)
    st.code("conda install -c conda-forge pandas", language="bash")

    st.info("""
    **Alternative Installation:** Using pip (Manual Dependency Management)
    """)
    st.code("pip install pandas", language="bash")

    st.markdown("""
    **Why use Conda?**
    - Conda ensures compatibility with **NumPy, SciPy, and Matplotlib**.
    - Pip installation may require additional dependencies.
    """)

    # 📂 Loading Climate Data with Pandas
    st.markdown("### 📂 Loading Climate Data with Pandas")
    st.markdown("""
    **Pandas makes it easy to load and explore climate datasets stored in CSV format.**
    """)

    st.code("""
import pandas as pd

# Load a CSV climate dataset
df = pd.read_csv("climate_data.csv")

# Display first few rows
print(df.head())
    """, language="python")

    # 🔍 Exploring Climate Data
    st.markdown("### 🔍 Exploring Climate Data")
    st.markdown("""
    Let's check the dataset's **summary, missing values, and basic statistics**.
    """)

    st.code("""
# Display summary of dataset
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Get summary statistics
print(df.describe())
    """, language="python")

    # 📅 Handling Time-Series Data in Pandas
    st.markdown("### 📅 Handling Time-Series Data in Pandas")
    st.markdown("""
    **Time is a critical axis in climate data.** Pandas provides powerful datetime functionality.
    """)

    st.code("""
# Convert column to datetime format
df["date"] = pd.to_datetime(df["date"])

# Set datetime as index
df.set_index("date", inplace=True)

# Display first few rows
print(df.head())
    """, language="python")

    # 📈 Resampling Climate Data (Daily to Monthly)
    st.markdown("### 📈 Resampling Climate Data (Daily to Monthly)")
    st.markdown("""
    Resampling is useful for **aggregating climate data** at different time scales.
    """)

    st.code("""
# Resample daily data to monthly averages
monthly_avg = df.resample("M").mean()

# Display first few rows
print(monthly_avg.head())
    """, language="python")

    # 🔍 Filtering Climate Data
    st.markdown("### 🔍 Filtering Climate Data")
    st.markdown("""
    Pandas allows **easy filtering of climate data** for specific conditions.
    """)

    st.code("""
# Filter data for temperatures above 30°C
hot_days = df[df["temperature"] > 30]

# Display first few rows
print(hot_days.head())
    """, language="python")

    # 📊 Visualizing Time-Series Climate Data
    st.markdown("### 📊 Visualizing Time-Series Climate Data")
    st.markdown("""
    **Pandas integrates with Matplotlib** for quick and easy visualization.
    """)

    st.code("""
import matplotlib.pyplot as plt

# Plot temperature trends over time
df["temperature"].plot(figsize=(10, 5), title="Temperature Trend Over Time")
plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.grid()
plt.show()
    """, language="python")

    # 🚀 Advanced: Rolling Window for Climate Data Trends
    st.markdown("### 🚀 Advanced: Rolling Window for Climate Data Trends")
    st.markdown("""
    Rolling averages help **smooth climate data** and reveal trends.
    """)

    st.code("""
# Compute rolling average for temperature
df["temperature_rolling"] = df["temperature"].rolling(window=7).mean()

# Plot rolling average trend
df[["temperature", "temperature_rolling"]].plot(figsize=(10, 5), title="7-Day Rolling Average Temperature")
plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.grid()
plt.show()
    """, language="python")

    st.success("✅ **Pandas makes climate data management efficient and easy!**")


# -------------------------------------------------
# 📈 SciPy - Scientific Computing for Climate Analysis
# -------------------------------------------------
def show_scipy():
    st.subheader("📈 SciPy - Scientific Computing for Climate Analysis")

    st.markdown("""
    **Why Use SciPy?**
    - SciPy extends NumPy and provides additional functionality for **scientific computing**.
    - Used for **interpolation, signal processing, and optimization** in climate science.
    - Contains statistical tools for **regression analysis, curve fitting, and data smoothing**.
    """)

    # 🔧 Installation Instructions
    st.markdown("### 🔧 Installation Methods")
    st.info("""
    **Preferred Installation:** Using Conda (Ensures Compatibility)
    """)
    st.code("conda install -c conda-forge scipy", language="bash")

    st.info("""
    **Alternative Installation:** Using pip (Manual Dependency Management)
    """)
    st.code("pip install scipy", language="bash")

    st.markdown("""
    **Why use Conda?**
    - Conda ensures compatibility with **NumPy and Matplotlib**.
    - Pip installation may require additional dependencies.
    """)

    # 📈 Linear Regression: Temperature Trends Over Time
    st.markdown("### 📈 Linear Regression: Temperature Trends Over Time")
    st.markdown("""
    Linear regression helps **detect long-term temperature trends** in climate data.
    """)

    st.code("""
import numpy as np
from scipy.stats import linregress

# Sample temperature data over years
years = np.array([2000, 2005, 2010, 2015, 2020])
temperature = np.array([14.2, 14.5, 14.8, 15.1, 15.5])

# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(years, temperature)

# Display results
print(f"Temperature Trend: {slope:.3f}°C per year")
print(f"R-squared Value: {r_value**2:.3f}")
    """, language="python")

    # 🔍 Interpolation: Filling Missing Climate Data
    st.markdown("### 🔍 Interpolation: Filling Missing Climate Data")
    st.markdown("""
    Climate datasets often have **missing values**. SciPy provides interpolation techniques to fill them.
    """)

    st.code("""
import numpy as np
from scipy.interpolate import interp1d

# Given years and corresponding temperature data (some missing)
years = np.array([2000, 2005, 2010, 2020])
temperature = np.array([14.2, 14.5, np.nan, 15.5])

# Interpolation function
interpolator = interp1d(years[~np.isnan(temperature)], temperature[~np.isnan(temperature)], kind="linear", fill_value="extrapolate")

# Fill missing value
filled_temperature = interpolator(years)
print(f"Interpolated Temperature: {filled_temperature}")
    """, language="python")

    # 🔎 Smoothing Climate Data with Signal Processing
    st.markdown("### 🔎 Smoothing Climate Data with Signal Processing")
    st.markdown("""
    SciPy's **signal processing tools** help smooth noisy climate data for better trend analysis.
    """)

    st.code("""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Simulated noisy temperature data
time = np.arange(0, 100, 1)
temperature = np.sin(time * 0.1) + np.random.normal(0, 0.2, len(time))

# Apply Savitzky-Golay filter to smooth data
smoothed_temp = savgol_filter(temperature, window_length=11, polyorder=2)

# Plot original and smoothed temperature data
plt.figure(figsize=(8,5))
plt.plot(time, temperature, color="gray", alpha=0.5, label="Noisy Data")
plt.plot(time, smoothed_temp, color="red", label="Smoothed Data")
plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.title("Smoothed Temperature Trends")
plt.legend()
plt.grid()
plt.show()
    """, language="python")

    # 📊 Curve Fitting: Estimating Climate Trends
    st.markdown("### 📊 Curve Fitting: Estimating Climate Trends")
    st.markdown("""
    Curve fitting is useful for estimating **non-linear trends in climate data**.
    """)

    st.code("""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define an exponential growth function
def climate_model(x, a, b, c):
    return a * np.exp(b * (x - 2000)) + c

# Simulated climate data
years = np.array([2000, 2005, 2010, 2015, 2020])
temperature = np.array([14.2, 14.5, 14.8, 15.1, 15.5])

# Fit the curve
params, covariance = curve_fit(climate_model, years, temperature)

# Generate smooth curve
future_years = np.linspace(2000, 2030, 100)
fitted_temps = climate_model(future_years, *params)

# Plot results
plt.figure(figsize=(8,5))
plt.scatter(years, temperature, color="blue", label="Observed Data")
plt.plot(future_years, fitted_temps, color="red", linestyle="--", label="Fitted Curve")
plt.xlabel("Year")
plt.ylabel("Temperature (°C)")
plt.title("Fitted Temperature Trend")
plt.legend()
plt.grid()
plt.show()
    """, language="python")

    st.success("✅ **SciPy provides powerful tools for scientific computing in climate science!**")


# -------------------------------------------------
# 🌎 Cartopy - Geospatial Mapping for Climate Science
# -------------------------------------------------
def show_cartopy():
    st.subheader("🌎 Cartopy - Geospatial Analysis for Climate Science")

    st.markdown("""
    **Why Use Cartopy?**
    - Cartopy enables **map projections** for accurate climate data visualization.
    - Supports plotting of **temperature distribution, wind patterns, and storm tracking**.
    - Provides **customized geographic features** like coastlines, political boundaries, and terrain.
    - Works seamlessly with **Matplotlib** for high-quality climate maps.
    """)

    # 🔧 Installation Instructions
    st.markdown("### 🔧 Installation Methods")
    st.info("""
    **Preferred Installation:** Using Conda (Ensures Compatibility)
    """)
    st.code("conda install -c conda-forge cartopy", language="bash")

    st.info("""
    **Alternative Installation:** Using pip (Manual Dependency Management)
    """)
    st.code("pip install cartopy", language="bash")

    st.markdown("""
    **Why use Conda?**
    - Conda ensures compatibility with **Matplotlib and PROJ4 (cartographic projections)**.
    - Pip installation may require additional dependencies.
    """)

    # 📍 Basic Map with Cartopy
    st.markdown("### 📍 Basic Map with Cartopy")
    st.markdown("""
    Cartopy allows the creation of **customized world maps** with coastlines.
    """)

    st.code("""
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

# Create a basic map
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines()
plt.title('Basic Map with Cartopy')
plt.show()
    """, language="python")

    # 🗺️ Understanding Map Projections
    st.markdown("### 🗺️ Understanding Map Projections")
    st.markdown("""
    **Why are map projections important?**
    - Climate data is stored in different coordinate systems (lat/lon, polar, UTM).
    - Cartopy allows projection transformations for **accurate geospatial visualization**.
    - Common projections:
      - `PlateCarree()`: Default lat/lon projection.
      - `Robinson()`: Used for global temperature maps.
      - `LambertConformal()`: Best for regional climate data.
    """)

    st.code("""
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

# Create a map with different projections
fig, axs = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': ccrs.Robinson()})

# Plot world map with coastlines
for ax in axs:
    ax.coastlines()
    ax.set_global()

axs[0].set_title("Robinson Projection")
axs[1].set_title("PlateCarree Projection")

plt.show()
    """, language="python")

    # 📊 Visualizing Temperature Data on a Map
    st.markdown("### 📊 Visualizing Temperature Data on a Map")
    st.markdown("""
    Cartopy allows overlaying **climate datasets (NetCDF, CSV, shapefiles)** onto a map.
    """)

    st.code("""
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Create random temperature data
lon = np.linspace(-180, 180, 100)
lat = np.linspace(-90, 90, 50)
temperature = np.random.rand(50, 100) * 30  # Simulating temperature data

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# Plot the temperature data as an image
ax.imshow(temperature, extent=[-180, 180, -90, 90], transform=ccrs.PlateCarree(), cmap="coolwarm")

ax.coastlines()
plt.title("Simulated Temperature Distribution")
plt.show()
    """, language="python")

    # 🌍 Adding Geographic Features
    st.markdown("### 🌍 Adding Geographic Features")
    st.markdown("""
    Cartopy allows **beautification** by adding:
    - Coastlines, country borders, rivers, and gridlines.
    - Custom color maps for **better visualization**.
    """)

    st.code("""
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# Add coastlines, country borders, and gridlines
ax.add_feature(cfeature.COASTLINE, linewidth=1.0)
ax.add_feature(cfeature.BORDERS, linestyle=":")

# Add gridlines
ax.gridlines(draw_labels=True)

plt.title("Cartopy with Geographic Features")
plt.show()
    """, language="python")

    # 📡 Wind Data Visualization
    st.markdown("### 📡 Wind Data Visualization with Arrows")
    st.markdown("""
    Wind patterns can be represented using **quiver (arrow) plots**.
    """)

    st.code("""
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Generate sample wind data
lon = np.linspace(-180, 180, 10)
lat = np.linspace(-90, 90, 5)
lon_grid, lat_grid = np.meshgrid(lon, lat)

u_wind = np.random.uniform(-10, 10, size=lon_grid.shape)
v_wind = np.random.uniform(-10, 10, size=lat_grid.shape)

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# Plot wind arrows
ax.quiver(lon_grid, lat_grid, u_wind, v_wind, transform=ccrs.PlateCarree())

ax.coastlines()
plt.title("Wind Flow Visualization")
plt.show()
    """, language="python")

    st.success("✅ **Cartopy is essential for geospatial climate data visualization!**")



# st.success("🎯 **You're now familiar with the essential Python libraries for climate data analysis!**")

