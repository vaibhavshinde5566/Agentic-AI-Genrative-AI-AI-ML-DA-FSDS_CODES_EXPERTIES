import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Movie Rating Analytics",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a "beautiful" look
st.markdown("""
<style>
    /* Main background and text colors */
    .stApp {
        background-color: #f8f9fa;
        color: #333333;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        color: #2c3e50;
        font-weight: 600;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #2c3e50;
        color: #ffffff;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #ecf0f1;
    }
    
    /* Custom container styling */
    .metric-container {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    /* Plot containers */
    .plot-container {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        # Assuming the file is in the same directory
        movies = pd.read_csv('Movie-Rating.csv')
        
        # Rename columns as per the notebook analysis
        movies.columns = ['Film', 'Genre', 'CriticRating', 'AudienceRatings', 'BudgetMillions', 'Year']
        
        # Convert to category types for efficiency and correct plotting behavior
        movies.Film = movies.Film.astype('category')
        movies.Genre = movies.Genre.astype('category')
        movies.Year = movies.Year.astype('category')
        
        return movies
    except FileNotFoundError:
        st.error("Error: 'Movie-Rating.csv' not found. Please ensure the file is in the app directory.")
        return None

movies = load_data()

# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------
st.sidebar.title("🎬 Visualization Gallery")
st.sidebar.markdown("Explore movie ratings, budgets, and trends.")

options = [
    "Dataset Overview",
    "Joint Plot: Critic vs Audience",
    "Distributions: Histograms",
    "Genre Analysis: Box & Violin",
    "KDE Analysis: Rating Density",
    "Facet Grid: Trends by Genre"
]

selection = st.sidebar.radio("Choose a Plot Type", options)

st.sidebar.markdown("---")
st.sidebar.info("Data source: Movie-Rating.csv")

# -----------------------------------------------------------------------------
# Main Content
# -----------------------------------------------------------------------------

if movies is not None:
    st.title("🎥 Movie Rating Advanced Analytics")
    st.markdown("### Exploratory Data Analysis with Seaborn & Streamlit")

    if selection == "Dataset Overview":
        st.header("📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Movies", len(movies))
        with col2:
            st.metric("Unique Genres", len(movies.Genre.unique()))
        with col3:
            st.metric("Avg Budget ($M)", f"{movies.BudgetMillions.mean():.1f}")
        with col4:
            st.metric("Avg Critic Rating", f"{movies.CriticRating.mean():.1f}%")

        st.subheader("Data Preview")
        st.dataframe(movies.head(10), use_container_width=True)
        
        st.subheader("Statistical Summary")
        st.dataframe(movies.describe(), use_container_width=True)

    elif selection == "Joint Plot: Critic vs Audience":
        st.header("🔗 Joint Plot: Critic vs Audience Ratings")
        st.markdown("Analyze the relationship between Critic Ratings and Audience Ratings.")
        
        kind = st.selectbox("Select Plot Kind", ["scatter", "hex", "kde", "reg"], index=0)
        
        fig = sns.jointplot(data=movies, x='CriticRating', y='AudienceRatings', kind=kind, height=8, color="#3498db")
        st.pyplot(fig)

    elif selection == "Distributions: Histograms":
        st.header("📈 Distributions")
        
        plot_type = st.selectbox("Select Variable to Visualize", ["BudgetMillions", "CriticRating", "AudienceRatings"])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=movies, x=plot_type, kde=True, bins=20, color="#e74c3c", ax=ax)
        ax.set_title(f"Distribution of {plot_type}", fontsize=15)
        st.pyplot(fig)
        
        # Stacked Histogram
        st.subheader("Stacked Histogram by Genre")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        # Filter top genres to avoid clutter if needed, or show all
        sns.histplot(data=movies, x=plot_type, hue='Genre', multiple="stack", ax=ax2)
        ax2.set_title(f"{plot_type} Distribution by Genre", fontsize=15)
        st.pyplot(fig2)

    elif selection == "Genre Analysis: Box & Violin":
        st.header("🎭 Genre Analysis")
        
        viz_type = st.radio("Select Visualization", ["Box Plot", "Violin Plot"], horizontal=True)
        
        if viz_type == "Box Plot":
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.boxplot(data=movies, x='Genre', y='CriticRating', palette="viridis", ax=ax)
            ax.set_title("Critic Ratings by Genre (Box Plot)", fontsize=15)
            st.pyplot(fig)
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.violinplot(data=movies, x='Genre', y='CriticRating', palette="magma", ax=ax)
            ax.set_title("Critic Ratings by Genre (Violin Plot)", fontsize=15)
            st.pyplot(fig)

    elif selection == "KDE Analysis: Rating Density":
        st.header("🔥 KDE Analysis: Rating Density")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.kdeplot(data=movies, x='CriticRating', y='AudienceRatings', cmap="Reds", shade=True, shade_lowest=False, ax=ax)
        # Overlay scatter for better context
        sns.kdeplot(data=movies, x='CriticRating', y='AudienceRatings', cmap="Reds", ax=ax)
        ax.set_title("Kernel Density Estimate: Critic vs Audience Ratings", fontsize=15)
        st.pyplot(fig)

    elif selection == "Facet Grid: Trends by Genre":
        st.header("🧩 Facet Grid: Trends by Genre")
        st.markdown("Scatter plot of Critic vs Audience Ratings, faceted by Genre.")
        
        # FacetGrid is a bit tricky in Streamlit as it creates its own figure
        g = sns.FacetGrid(movies, row='Genre', hue='Genre', aspect=4, height=2, palette="coolwarm")
        g.map(plt.scatter, 'CriticRating', 'AudienceRatings', alpha=0.7)
        g.add_legend()
        
        st.pyplot(g.fig)

else:
    st.warning("Please upload 'Movie-Rating.csv' to the application directory.")
