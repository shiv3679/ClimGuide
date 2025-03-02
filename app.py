import streamlit as st

# Page Config
st.set_page_config(page_title="Climate Learning Platform", page_icon="🌍", layout="wide")


from sections.home import show_home
from sections.python_basics import show_python_basics
from sections.netcdf_intro import show_netcdf_intro
from sections.visualization import show_visualization
from sections.playground import show_playground



# Custom CSS for Styling & Animations
st.markdown("""
    <style>
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #002B5B;
        }
        
        /* Sidebar Titles */
        .sidebar-title {
            color: #ffffff;
            font-size: 18px;
            font-weight: bold;
            text-align: center;
        }

        /* Fade-in effect */
        .fade-in {
            animation: fadeIn 0.6s ease-in-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        /* Button Styling */
        .start-button {
            background-color: #0077b6;
            color: white;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 8px;
            text-align: center;
            font-size: 18px;
            display: block;
            margin: auto;
            transition: 0.3s ease-in-out;
        }
        
        .start-button:hover {
            background-color: #0096c7;
        }

    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.markdown('<p class="sidebar-title">📚 Climate Learning Chapters</p>', unsafe_allow_html=True)
chapters = {
    "🏠 Home": show_home,
    "🐍 Python Basics": show_python_basics,
    "📂 Understanding NetCDF": show_netcdf_intro,
    "📊 Visualizing Climate Data": show_visualization,
    "🛠 Playground": show_playground
}

# Sidebar Chapter Selection
selected_chapter = st.sidebar.radio("📖 Select a chapter:", list(chapters.keys()))

# Load the selected chapter
st.markdown('<div class="fade-in">', unsafe_allow_html=True)
chapters[selected_chapter]()  
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.markdown("**Developed by Shiv Shankar Singh**", unsafe_allow_html=True)
