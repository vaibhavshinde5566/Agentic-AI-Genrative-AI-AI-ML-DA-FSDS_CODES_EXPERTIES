import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="Hospital Data Analysis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h2, h3 {
        color: #34495e;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stButton>button {
        color: white;
        background-color: #e74c3c;
        border-radius: 10px;
        border: none;
        padding: 10px 24px;
        font-size: 16px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #c0392b;
        color: white;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Data Loading
@st.cache_data
def load_data():
    try:
        # Assuming heart.csv is in the same directory
        df = pd.read_csv("heart.csv")
        return df
    except FileNotFoundError:
        st.error("File 'heart.csv' not found. Please ensure it is in the same directory.")
        return None

df = load_data()

if df is not None:
    # Sidebar
    st.sidebar.header("Filter Options")
    
    # Sex Filter
    sex_options = {0: "Female", 1: "Male"}
    selected_sex = st.sidebar.multiselect(
        "Select Sex",
        options=list(sex_options.keys()),
        format_func=lambda x: sex_options[x],
        default=list(sex_options.keys())
    )
    
    # Age Filter
    min_age = int(df['age'].min())
    max_age = int(df['age'].max())
    selected_age = st.sidebar.slider(
        "Select Age Range",
        min_age, max_age,
        (min_age, max_age)
    )
    
    # Filter Data
    filtered_df = df[
        (df['sex'].isin(selected_sex)) &
        (df['age'].between(selected_age[0], selected_age[1]))
    ]

    # Main Content
    st.title("🏥 Hospital Data Analysis Dashboard")
    st.markdown("### Heart Disease Analysis & Visualization")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Patients", len(filtered_df))
    with col2:
        st.metric("Avg Age", f"{filtered_df['age'].mean():.1f}")
    with col3:
        st.metric("Avg Cholesterol", f"{filtered_df['chol'].mean():.1f}")
    with col4:
        st.metric("Heart Disease Cases", filtered_df[filtered_df['target'] == 1].shape[0])

    st.markdown("---")

    # Visualizations
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Heart Disease Distribution")
        fig, ax = plt.subplots(figsize=(8, 6))
        # Custom palette
        colors = ["#2ecc71", "#e74c3c"] # Green for No, Red for Yes
        sns.countplot(x='target', data=filtered_df, palette=colors, ax=ax)
        ax.set_xticklabels(['No Disease', 'Disease'])
        ax.set_xlabel("Diagnosis")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    with col_right:
        st.subheader("Sex vs Heart Disease")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.countplot(x='sex', hue='target', data=filtered_df, palette="Set2", ax=ax2)
        ax2.set_xticklabels(['Female', 'Male'])
        ax2.set_xlabel("Sex")
        ax2.set_ylabel("Count")
        ax2.legend(title='Disease', labels=['No', 'Yes'])
        st.pyplot(fig2)

    st.markdown("---")
    
    col_bottom_1, col_bottom_2 = st.columns(2)
    
    with col_bottom_1:
        st.subheader("Age Distribution")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.histplot(data=filtered_df, x='age', hue='target', kde=True, element="step", palette="coolwarm", ax=ax3)
        ax3.set_title("Age Distribution by Disease Status")
        st.pyplot(fig3)
        
    with col_bottom_2:
        st.subheader("Chest Pain Type vs Target")
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        sns.countplot(x='cp', hue='target', data=filtered_df, palette="pastel", ax=ax4)
        ax4.set_xlabel("Chest Pain Type")
        ax4.set_ylabel("Count")
        ax4.legend(title='Disease', labels=['No', 'Yes'])
        st.pyplot(fig4)

    # Correlation Heatmap (Optional toggle)
    if st.checkbox("Show Correlation Heatmap"):
        st.subheader("Correlation Heatmap")
        fig5, ax5 = plt.subplots(figsize=(12, 10))
        corr = filtered_df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax5)
        st.pyplot(fig5)

else:
    st.info("Awaiting data... Please ensure 'heart.csv' is present.")
