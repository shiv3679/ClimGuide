import streamlit as st
import xarray as xr
import matplotlib.pyplot as plt

# File path
LOCAL_FILE = "datasets/air.mon.ltm.nc"
DOWNLOAD_URL = "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/Monthlies/air.mon.ltm.nc"

def show_netcdf_intro():
    st.title("📂 NetCDF Files & Climate Data Analysis")

    st.markdown("""
    **What is NetCDF?**  
    - NetCDF (**Network Common Data Form**) is the standard format for **climate datasets**.
    - Stores **multi-dimensional data** (e.g., temperature over time, latitude, longitude).
    - Used by **NASA, NOAA, IPCC, and climate scientists worldwide**.

    **Dataset Used**:  
    - 📁 `air.mon.ltm.nc`: Long-term mean **monthly air temperature**.

    **🔗 [Download Latest NetCDF File]({DOWNLOAD_URL}) (NOAA Reanalysis)**
    """)

    # Step 1: Load NetCDF Dataset
    st.markdown("### 📌 Step 1: Load the NetCDF Dataset")
    st.code("""
import xarray as xr

# Open the dataset
file_path = "datasets/air.mon.ltm.nc"
data = xr.open_dataset(file_path)

# Display dataset interactively
data
    """, language="python")

    # Load Dataset
    try:
        data = xr.open_dataset(LOCAL_FILE)
        st.success("✅ Dataset loaded successfully!")
    except FileNotFoundError:
        st.error("⚠️ NetCDF file not found! Please ensure `air.mon.ltm.nc` exists in `datasets/`.")
        return

    # Extract Air Temperature Variable
    st.markdown("### 📌 Step 2: Extract the `air` Variable")
    st.code("""
# Extract air variable (Temperature)
data = data['air']
print(data)
    """, language="python")

    data = data['air']
    st.text(data)

    # Step 3: Select a Season
    st.markdown("### 📌 Step 3: Select a Season (Summer or Winter)")
    season = st.radio("Choose a Season:", ["Summer (JJAS)", "Winter (DJF)"])

    # Define months for slicing
    if season == "Summer (JJAS)":
        season_months = [6, 7, 8, 9]  # June, July, August, September
    else:
        season_months = [12, 1, 2]  # December, January, February

    st.code(f"""
# Extract only the months corresponding to {season}
months = data.time.dt.month
data_season = data.sel(time=months.isin({season_months}))
    """, language="python")

    # Slice dataset for selected season
    months = data.time.dt.month
    data_season = data.sel(time=months.isin(season_months))

    # Step 4: Select a Climate Region
    st.markdown("### 📌 Step 4: Select a Climate Region")
    region = st.radio("Choose a Climate Zone:", ["Tropics (±30°)", "Temperate (30° to 60°)", "Polar (Above 60°)"])

    # Define latitude ranges
    if region == "Tropics (±30°)":
        data_region = data_season.where((data_season.lat >= -23.5) & (data_season.lat <= 23.5), drop=True)
    elif region == "Temperate (30° to 60°)":
        data_region = data_season.where(((data_season.lat >= 23.5) & (data_season.lat <= 66.5)) | 
                                        ((data_season.lat <= -23.5) & (data_season.lat >= -66.5)), drop=True)
    else:
        data_region = data_season.where((data_season.lat >= 66.5) | (data_season.lat <= -66.5), drop=True)

    st.code(f"""
# Extract data for {region}
data_region = data_season.where((condition for selected region), drop=True)
    """, language="python")

    # Compute mean across lat, lon, and time
    st.markdown("### 📌 Step 5: Compute Vertical Temperature Profile")
    st.code("""
# Compute mean over lat, lon, and time
data_levels = data_region.mean(dim=['lat', 'lon', 'time'])
print(data_levels)
    """, language="python")

    data_levels = data_region.mean(dim=['lat', 'lon', 'time'])
    st.text(data_levels)

    # Step 6: Interactive Latitude Selection
    st.markdown("### 📌 Step 6: Explore Vertical Profile at Different Latitudes")
    lat_value = st.slider("Select Latitude", float(data.lat.min()), float(data.lat.max()), float(data.lat.mean()))

    st.code(f"""
# Extract data at latitude = {lat_value}
data_lat = data_season.sel(lat={lat_value}, method="nearest")
    """, language="python")

    data_lat = data_season.sel(lat=lat_value, method="nearest")

    # Step 7: Plot Vertical Thermal Structure
    st.markdown("### 📌 Step 7: Compare Vertical Temperature Structures")
    st.code("""
import matplotlib.pyplot as plt

# Create vertical temperature structure plot
plt.plot(data_levels, data_levels.level, 'ro-', label="Region Mean")
plt.plot(data_lat.mean(dim=['lon', 'time']), data_lat.level, 'bo-', label=f"Lat: {lat_value}")
plt.ylim(max(data.level), min(data.level))  # Invert Y-axis
plt.xlabel('Temperature (K)')
plt.ylabel('Pressure (hPa)')
plt.title(f'Vertical Thermal Structure of {region} during {season}')
plt.legend()
plt.show()
    """, language="python")

    # Generate Plot
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.plot(data_levels, data_levels.level, 'ro-', label="Region Mean")
    ax.plot(data_lat.mean(dim=['lon', 'time']), data_lat.level, 'bo-', label=f"Lat: {lat_value}")

    ax.set_ylim(max(data.level), min(data.level))  # Invert Y-axis
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Pressure (hPa)')
    ax.set_title(f'Vertical Thermal Structure of {region} during {season}')
    ax.legend()
    st.pyplot(fig)

    # Close Dataset
    data.close()

    # write the ending 
    st.markdown(" At this point, you’ve explored NetCDF files, visualized vertical temperature structures, and analyzed how seasons & latitudes impact climate data. This is exactly what climate scientists do when studying the Earth's atmosphere! 🌍✨**")
