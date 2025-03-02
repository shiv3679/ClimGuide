import streamlit as st
import xarray as xr
import matplotlib.pyplot as plt

# File path
LOCAL_FILE = "datasets/air.mon.ltm.nc"
DOWNLOAD_URL = "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/Monthlies/air.mon.ltm.nc"

def show_netcdf_intro():
    st.title("📂 Introduction to NetCDF Files in Climate Science")

    st.markdown("""
    **What is NetCDF?**  
    - NetCDF (**Network Common Data Form**) is the standard format for **climate data storage**.
    - It allows storing **multi-dimensional arrays** (e.g., temperature over time and space).
    - Used by **NASA, NOAA, IPCC, and climate researchers worldwide**.

    **Dataset Used**:  
    - 📁 `air.mon.ltm.nc` – Long-term mean **monthly air temperature**.
    """)

    # 🔗 Download Link
    st.markdown(f"📥 **[Download Latest NetCDF File]({DOWNLOAD_URL})** (NOAA Reanalysis)")

    # Step 1: Load NetCDF Dataset
    st.markdown("### 📌 Step 1: Load the NetCDF Dataset")
    st.code("""
import xarray as xr

# Open the dataset
file_path = "datasets/air.mon.ltm.nc"
ds = xr.open_dataset(file_path)

# Print dataset summary
print(ds)
    """, language="python")

    # Load Dataset
    try:
        ds = xr.open_dataset(LOCAL_FILE)
        st.success("✅ Dataset loaded successfully!")
    except FileNotFoundError:
        st.error("⚠️ NetCDF file not found! Please ensure `air.mon.ltm.nc` exists in `datasets/`.")
        return

    # Show Dataset Metadata
    st.markdown("### 📌 Step 2: View Dataset Metadata")
    st.code("""
# View dataset structure
print(ds)
    """, language="python")

    st.text(ds)

    # Show Variables
    st.markdown("### 📌 Step 3: View Available Variables")
    st.code("""
# List all variables in the dataset
print(ds.variables.keys())
    """, language="python")

    st.write(list(ds.variables))

    # Select a Variable
    st.markdown("### 📌 Step 4: Select a Variable to Explore")
    variable = st.selectbox("Choose a variable to analyze:", list(ds.variables))

    if variable:
        # Show Variable Metadata
        st.markdown("### 🔍 Step 5: View Metadata of Selected Variable")
        st.code(f"""
# Show metadata for {variable}
print(ds["{variable}"])
        """, language="python")

        st.text(ds[variable])

        # Handle Time Selection
        st.markdown("### 📌 Step 6: Select a Time Index (If Applicable)")
        if "time" in ds[variable].dims:
            time_index = st.slider("Select Time Index", 0, len(ds["time"]) - 1, 0)
            plot_data = ds[variable].isel(time=time_index)
            st.code(f"""
# Select a specific time index
data_at_time = ds["{variable}"].isel(time={time_index})
print(data_at_time)
            """, language="python")
        else:
            plot_data = ds[variable]

        # Plot the Data
        st.markdown("### 📌 Step 7: Visualizing the Variable")
        st.code(f"""
import matplotlib.pyplot as plt

# Plot the variable
fig, ax = plt.subplots(figsize=(8, 5))
plot_data.plot(ax=ax, cmap="coolwarm")
plt.title("{variable} at Selected Time Index")
plt.grid()
plt.show()
        """, language="python")

        fig, ax = plt.subplots(figsize=(8, 5))
        if "lat" in plot_data.dims and "lon" in plot_data.dims:
            plot_data.plot(ax=ax, cmap="coolwarm")
        else:
            plot_data.plot(ax=ax)

        plt.title(f"{variable} at Selected Time Index")
        plt.grid()
        st.pyplot(fig)

    # Close Dataset
    ds.close()
