import streamlit as st
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
from catboost import CatBoostRegressor

st.set_page_config(page_title="Airbnb Price Predictor", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    color: #ffffff !important;
}

.stApp {
    background-color: #0E1117 !important;
}

section[data-testid="stSidebar"] {
    background-color: #1E1E1E !important;
    border-right: 1px solid #2A2A2A !important;
}

.block-container {
    background-color: #1E1E1E !important;
    padding: 20px;
    border-radius: 12px;
}

h1, h2, h3 {
    color: #EAF6FF !important;
    font-weight: 600 !important;
}

.stNumberInput input,
div[data-baseweb="select"] > div {
    background-color: #2A2A2A !important;
    color: #ffffff !important;
    border: 1.5px solid #3A3A3A !important;
    border-radius: 6px !important;
}

.stButton>button {
    background-color: #e63946 !important;
    color: white !important;
    border-radius: 10px;
    border: none;
    padding: 14px 28px;
    width: 100%;
    font-size: 20px !important;
    transition: 0.2s;
}
.stButton>button:hover {
    background-color: #b71c1c !important;
    box-shadow: 0 0 12px #ff4d4d;
}

header[data-testid="stHeader"] { display: none !important; }
footer, #MainMenu {visibility: hidden !important;}
</style>
""", unsafe_allow_html=True)


df_clean = pd.read_csv("airbnb_train.csv")

model = CatBoostRegressor()
model.load_model("catboost_airbnb.cbm")


cities = sorted(df_clean["city"].unique())
selected_city = st.selectbox("Select a city", cities)

st.subheader("Click on the map to select the Airbnb location")

city_data = df_clean[df_clean["city"] == selected_city]
city_center = [city_data["latitude"].mean(), city_data["longitude"].mean()]

m = folium.Map(
    location=city_center,
    zoom_start=13,
    tiles="CartoDB DarkMatter"
)


if "price" in city_data.columns:
    prices = city_data["price"]
    q1, q2 = prices.quantile(0.33), prices.quantile(0.66)

    for _, row in city_data.iterrows():
        if row["price"] <= q1:
            color = "#66b3ff"
        elif row["price"] <= q2:
            color = "#1f78ff"
        else:
            color = "#003070"

        folium.CircleMarker(
            [row["latitude"], row["longitude"]],
            radius=6,
            color=color,
            fill=True,
            fill_opacity=0.9
        ).add_to(m)

map_data = st_folium(m, width=1600, height=800)

lat, lon = None, None
if map_data and map_data.get("last_clicked"):
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]
    st.info(f"Selected location → {lat:.5f}, {lon:.5f}")
else:
    st.warning("Click a location on the map")


st.subheader("Housing characteristics")

accommodates = st.number_input("Number of guests", 1, 20, 2)
bedrooms = st.number_input("Bedrooms", 0, 10, 1)
bathrooms = st.number_input("Bathrooms", 0, 10, 1)

room_type = st.selectbox("Room type", df_clean["room_type"].unique())
property_type = st.selectbox("Property type", df_clean["property_type"].unique())

train_cols = [
    'host_since', 'host_response_rate', 'host_acceptance_rate',
    'host_listings_count', 'host_total_listings_count',
    'latitude', 'longitude', 'accommodates', 'bathrooms', 'bedrooms', 'beds',
    'availability_30', 'availability_60', 'availability_90', 'availability_365',
    'number_of_reviews', 'number_of_reviews_ltm', 'number_of_reviews_l30d',
    'first_review', 'last_review', 'reviews_per_month',
    'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
    'review_scores_checkin', 'review_scores_communication',
    'review_scores_location', 'review_scores_value',
    'host_response_time', 'host_is_superhost', 'host_has_profile_pic',
    'host_identity_verified', 'room_type', 'has_availability', 'city',
    'property_type', 'verification_group',
    'bathrooms_filled_by_median', 'bedrooms_filled_by_median', 'beds_filled_by_median',
    'host_response_rate_filled_by_median',
    'host_acceptance_rate_filled_by_median'
]


input_dict = {col: 0 for col in train_cols}


input_dict["latitude"] = lat
input_dict["longitude"] = lon
input_dict["accommodates"] = accommodates
input_dict["bathrooms"] = bathrooms
input_dict["bedrooms"] = bedrooms

input_dict["city"] = selected_city
input_dict["room_type"] = room_type
input_dict["property_type"] = property_type


for col in [
    "host_response_time", "host_is_superhost", "host_has_profile_pic",
    "host_identity_verified", "has_availability", "verification_group"
]:
    input_dict[col] = "Unknown"


input_df = pd.DataFrame([input_dict])[train_cols]  

if st.button("Predict price"):
    if lat is None or lon is None:
        st.error("Please click on the map first.")
    else:
        y_log = model.predict(input_df)[0]
        price = np.expm1(y_log)  
        st.success(f"💰 Predicted price: **{price:.2f} €**")
        st.balloons()



