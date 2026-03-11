import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# try optional plotting library
try:
    import plotly.express as px
    _USE_PLOTLY = True
except ImportError:
    _USE_PLOTLY = False

# base directory so paths work from anywhere
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "combined_airbnb.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "airbnb_price_model.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "models", "model_columns.pkl")

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"Data file not found: {DATA_PATH}")
        return pd.DataFrame()
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_columns():
    with open(COLUMNS_PATH, "rb") as f:
        return pickle.load(f)

# ---------- load everything ----------

data = load_data()
model = load_model()
model_columns = load_columns()

st.title("🏠 Airbnb Price Prediction Dashboard")

# navigation
page = st.sidebar.selectbox("Page", ["Overview", "Data", "Predict"])

if page == "Overview":
    st.header("Dataset Overview")
    if not data.empty:
        st.metric("Total listings", len(data))
        st.metric("Average price", f"${data['price'].mean():.2f}")

        if _USE_PLOTLY:
            price_hist = px.histogram(data, x="price", nbins=50, title="Price Distribution")
            st.plotly_chart(price_hist, width='stretch')

            counts = data['city'].value_counts().reset_index()
            counts.columns = ['city', 'count']
            city_bar = px.bar(counts, x='city', y='count', title='Listings by City')
            st.plotly_chart(city_bar, width='stretch')
        else:
            st.write("Install `plotly` to see interactive charts.")
    else:
        st.write("No data available.")

elif page == "Data":
    st.header("Raw Data Explorer")
    if not data.empty:
        city_filter = st.multiselect("Filter by city", sorted(data['city'].unique()))
        room_filter = st.multiselect("Filter by room type", sorted(data['room_type'].unique()))
        price_range = st.slider("Price range",
                                int(data['price'].min()),
                                int(data['price'].max()),
                                (int(data['price'].min()), int(data['price'].max())))
        df = data.copy()
        if city_filter:
            df = df[df['city'].isin(city_filter)]
        if room_filter:
            df = df[df['room_type'].isin(room_filter)]
        df = df[(df['price'] >= price_range[0]) & (df['price'] <= price_range[1])]
        st.dataframe(df)
    else:
        st.write("Dataset not loaded.")

elif page == "Predict":
    st.header("Price Estimator")
    st.sidebar.subheader("Listing Features")

    accommodates = st.sidebar.number_input("Accommodates", 1, 10, 2)
    bedrooms = st.sidebar.number_input("Bedrooms", 0, 10, 1)
    bathrooms = st.sidebar.number_input("Bathrooms", 0.0, 5.0, 1.0)
    beds = st.sidebar.number_input("Beds", 1, 10, 1)

    room_type = st.sidebar.selectbox(
        "Room Type",
        ["Entire home/apt", "Private room", "Shared room"]
    )

    city = st.sidebar.selectbox(
        "City",
        ["Boston", "Los Angeles", "San Francisco"]
    )

    # build input vector
    input_dict = {col: 0 for col in model_columns}
    for key, val in [
        ("accommodates", accommodates),
        ("bedrooms", bedrooms),
        ("bathrooms", bathrooms),
        ("beds", beds),
    ]:
        if key in input_dict:
            input_dict[key] = val
    rt_col = f"room_type_{room_type}"
    city_col = f"city_{city}"
    if rt_col in input_dict:
        input_dict[rt_col] = 1
    if city_col in input_dict:
        input_dict[city_col] = 1

    input_df = pd.DataFrame([input_dict])

    if st.button("Predict"):
        try:
            pred = model.predict(input_df)[0]
            st.success(f"Estimated Nightly Price: ${pred:.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.markdown("---")
    st.subheader("Quick Look at Data")
    if not data.empty:
        st.write(data.sample(5))

