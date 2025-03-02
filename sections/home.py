import streamlit as st

def show_home():
    st.markdown('<h1 style="text-align: center; color: #0077b6;">🌍 Climate Learning Platform</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 18px;">An interactive way to learn climate data science with Python and NetCDF</p>', unsafe_allow_html=True)


    st.image("assets/ice-logo.svg", use_container_width=True)


    # Introduction
    st.markdown("### 📌 What is this platform about?")
    st.markdown("""
    - This platform is designed to **teach students about climate data science** using Python.
    - Learn how to **work with NetCDF files, analyze trends, and visualize climate data**.
    - No prior experience required – everything is **interactive and beginner-friendly**! 🚀
    """)

    # Features Overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🐍 **Learn Python**\n\nStep-by-step tutorials for climate analysis.")
    with col2:
        st.success("📊 **Visualize Climate Data**\n\nExplore trends, maps, and geospatial patterns.")
    with col3:
        st.warning("📂 **Work with NetCDF**\n\nUpload, process, and analyze real climate datasets.")

    # Call to Action Button
    st.markdown('<a class="start-button" href="#">Start Learning 🚀</a>', unsafe_allow_html=True)