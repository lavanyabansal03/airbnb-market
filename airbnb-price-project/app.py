import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Airbnb Analytics",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Optional plotting library
try:
    import plotly.express as px
    _USE_PLOTLY = True
except ImportError:
    _USE_PLOTLY = False

# --- ASSETS & DATA ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "combined_airbnb.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "airbnb_price_model.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "models", "model_columns.pkl")

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH): return pd.DataFrame()
    return pd.read_csv(DATA_PATH)

@st.cache_resource # Use resource for models
def load_model():
    with open(MODEL_PATH, "rb") as f: return pickle.load(f)

@st.cache_data
def load_columns():
    with open(COLUMNS_PATH, "rb") as f: return pickle.load(f)

# --- LOAD ---
data = load_data()
try:
    model = load_model()
    model_columns = load_columns()
except:
    model, model_columns = None, None

# --- SIDEBAR NAVIGATION ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/6/69/Airbnb_Logo_Bélo.svg", width=150)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📈 Market Overview", "🔍 Data Explorer", "💰 Price Predictor"])

# --- PAGE 1: MARKET OVERVIEW ---
if page == "📈 Market Overview":
    st.title("🏠 Airbnb Market Insights Overview")
    st.markdown("A comprehensive look at pricing and inventory across the market.")
    st.markdown("---")

    if not data.empty:
        # Top Row Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Listings", f"{len(data):,}")
        m2.metric("Avg. Nightly Price", f"${data['price'].mean():.2f}")
        m3.metric("Cities Covered", data['city'].nunique())
        m4.metric("Avg. Bedrooms", f"{data['bedrooms'].mean():.1f}")

        st.write("##")

        if _USE_PLOTLY:
            # --- ROW 1 ---
            r1c1, r1c2, r1c3 = st.columns(3)
            
            with r1c1:
                # 1. Price Distribution by City
                fig1 = px.box(data, x="city", y="price", color="city",
                             title="Price by City", template="plotly_white",
                             color_discrete_sequence=px.colors.qualitative.Prism)
                st.plotly_chart(fig1, use_container_width=True)
            
            with r1c2:
                # 2. Avg Price by Room Type
                avg_rt = data.groupby('room_type')['price'].mean().reset_index()
                fig2 = px.bar(avg_rt, x='room_type', y='price', color='room_type',
                             title="Avg Price: Room Type", template="plotly_white", text_auto='.0f')
                st.plotly_chart(fig2, use_container_width=True)

            with r1c3:
                # 3. Price Density
                fig3 = px.histogram(data, x="price", nbins=30, title="Price Density",
                                   color_discrete_sequence=['#FF5A5F'], template="plotly_white")
                st.plotly_chart(fig3, use_container_width=True)

            # --- ROW 2 ---
            r2c2, r2c3 = st.columns(2)

            with r2c2:
                # 5. Inventory Share (Donut)
                counts = data['room_type'].value_counts().reset_index()
                fig5 = px.pie(counts, values='count', names='room_type', hole=0.5,
                             title="Inventory Share", template="plotly_white",
                             color_discrete_sequence=px.colors.sequential.RdBu)
                st.plotly_chart(fig5, use_container_width=True)

            with r2c3:
                # 6. Avg Price by Accommodates
                avg_acc = data.groupby('accommodates')['price'].mean().reset_index()
                fig6 = px.line(avg_acc, x='accommodates', y='price', markers=True,
                              title="Price by Capacity", template="plotly_white",
                              color_discrete_sequence=['#FF5A5F'])
                st.plotly_chart(fig6, use_container_width=True)

            # # Correlation Heatmap remains full-width below the grid
            # st.write("##")
            # st.subheader("🔗 Feature Correlation Matrix")
            # corr = data.select_dtypes(include=[np.number]).corr()
            # fig_heatmap = px.imshow(corr, text_auto=".2f", aspect="auto",
            #                        color_continuous_scale='RdBu_r', template="plotly_white")
            # st.plotly_chart(fig_heatmap, use_container_width=True)

    else:
        st.error("Data missing. Please check your file paths.")
# --- PAGE 2: DATA EXPLORER ---
elif page == "🔍 Data Explorer":
    st.title("🔍 Raw Data Explorer")
    
    with st.expander("Filter Options", expanded=True):
        f1, f2 = st.columns(2)
        with f1:
            city_filter = st.multiselect("Select Cities", sorted(data['city'].unique()))
        with f2:
            room_filter = st.multiselect("Room Types", sorted(data['room_type'].unique()))
        
        price_range = st.slider("Price Range ($)", 
                               int(data['price'].min()), int(data['price'].max()), 
                               (int(data['price'].min()), int(data['price'].max())))

    df = data.copy()
    if city_filter: df = df[df['city'].isin(city_filter)]
    if room_filter: df = df[df['room_type'].isin(room_filter)]
    df = df[(df['price'] >= price_range[0]) & (df['price'] <= price_range[1])]

    st.dataframe(df, use_container_width=True, height=500)
    st.download_button("Download CSV", df.to_csv(index=False), "filtered_airbnb.csv", "text/csv")

# --- PAGE 3: PREDICTOR ---
elif page == "💰 Price Predictor":
    st.title("💰 Smart Price Estimator")
    st.markdown("Adjust listing details to get a machine-learning based price estimate.")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("Listing Details")
        city = st.selectbox("City", ["Boston", "Los Angeles", "San Francisco"])
        room_type = st.selectbox("Room Type", ["Entire home/apt", "Private room", "Shared room"])
        
        c1, c2 = st.columns(2)
        accommodates = c1.number_input("Accommodates", 1, 16, 2)
        bedrooms = c2.number_input("Bedrooms", 0, 10, 1)
        bathrooms = c1.number_input("Bathrooms", 0.0, 10.0, 1.0, step=0.5)
        beds = c2.number_input("Beds", 1, 20, 1)

    with col2:
        st.subheader("Estimation")
        st.info("The model calculates price based on historical trends in the selected city.")
        
        if st.button("Calculate Optimal Price", use_container_width=True):
            if model and model_columns:
                # build input vector
                input_dict = {col: 0 for col in model_columns}
                for key, val in [("accommodates", accommodates), ("bedrooms", bedrooms), 
                                 ("bathrooms", bathrooms), ("beds", beds)]:
                    if key in input_dict: input_dict[key] = val
                
                rt_col, city_col = f"room_type_{room_type}", f"city_{city}"
                if rt_col in input_dict: input_dict[rt_col] = 1
                if city_col in input_dict: input_dict[city_col] = 1

                pred = model.predict(pd.DataFrame([input_dict]))[0]
                
                st.metric(label="Predicted Nightly Rate", value=f"${pred:.2f}")
                st.balloons()
            else:
                st.warning("Model files missing. Prediction unavailable.")

    st.markdown("---")
    st.subheader("Comparable Listings")
    st.write(data[data['city'] == city].head(5))