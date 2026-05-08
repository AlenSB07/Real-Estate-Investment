import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏡",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.stApp {
    background-color: #0E1117;
    color: white;
}

h1,h2,h3,h4,h5,h6,p,label {
    color: white !important;
}

[data-testid="stSidebar"] {
    background-color: #161A23;
}

.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    border: none;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}

.metric-card {
    background-color: #1c2333;
    padding: 25px;
    border-radius: 12px;
    text-align: center;
}

.success-box {
    background-color: #163d2b;
    padding: 20px;
    border-radius: 10px;
    font-size: 20px;
    font-weight: bold;
}

.error-box {
    background-color: #3d1616;
    padding: 20px;
    border-radius: 10px;
    font-size: 20px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():

    df = pd.read_csv("india_housing_prices.csv")
    

    return df

df = load_data()

# ---------------- CLEAN COLUMN NAMES ----------------
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

# ---------------- REMOVE DUPLICATES ----------------
df.drop_duplicates(inplace=True)

# ---------------- FILL MISSING VALUES ----------------
df.fillna(0, inplace=True)

# ---------------- NUMERIC CONVERSION ----------------
numeric_cols = [
    'price_in_lakhs',
    'size_in_sqft',
    'nearby_schools',
    'nearby_hospitals',
    'bhk',
    'year_built'
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.fillna(0, inplace=True)

# ---------------- FEATURE ENGINEERING ----------------
df['size_in_sqft'] = df['size_in_sqft'].replace(0, 1)

df['price_per_sqft'] = (
    df['price_in_lakhs'] / df['size_in_sqft']
)

df['age_of_property'] = 2026 - df['year_built']

# ---------------- TRANSPORT SCORE ----------------
transport_map = {
    "poor": 2,
    "medium": 5,
    "high": 8
}

df['public_transport_accessibility'] = (
    df['public_transport_accessibility']
    .astype(str)
    .str.lower()
)

df['transport_score'] = (
    df['public_transport_accessibility']
    .map(transport_map)
    .fillna(5)
)

# ---------------- GOOD INVESTMENT ----------------
df['good_investment'] = np.where(
    (
        df['price_per_sqft']
        < df['price_per_sqft'].median()
    )
    &
    (
        df['nearby_schools']
        > df['nearby_schools'].median()
    ),
    1,
    0
)

# ---------------- FEATURES ----------------
features = [
    'bhk',
    'size_in_sqft',
    'nearby_schools',
    'nearby_hospitals',
    'transport_score',
    'age_of_property'
]

X = df[features]

y_class = df['good_investment']

y_reg = df['price_in_lakhs']

# ---------------- TRAIN MODELS ----------------
@st.cache_resource
def train_models(X, y_class, y_reg):

    clf = RandomForestClassifier(
        n_estimators=20,
        max_depth=5,
        random_state=42
    )

    reg = RandomForestRegressor(
        n_estimators=20,
        max_depth=5,
        random_state=42
    )

    clf.fit(X, y_class)

    reg.fit(X, y_reg)

    return clf, reg

clf, reg = train_models(X, y_class, y_reg)

# ---------------- DYNAMIC DROPDOWNS ----------------
city_list = sorted(
    df['city']
    .dropna()
    .astype(str)
    .unique()
)

ptype_list = sorted(
    df['property_type']
    .dropna()
    .astype(str)
    .unique()
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🏠 Property Details")

city = st.sidebar.selectbox(
    "City",
    city_list
)

ptype = st.sidebar.selectbox(
    "Property Type",
    ptype_list
)

bhk = st.sidebar.slider(
    "BHK",
    1,
    6,
    2
)

size = st.sidebar.slider(
    "Size (SqFt)",
    500,
    5000,
    1500
)

schools = st.sidebar.slider(
    "Nearby Schools",
    0,
    10,
    3
)

hospitals = st.sidebar.slider(
    "Nearby Hospitals",
    0,
    10,
    2
)

transport = st.sidebar.selectbox(
    "Transport Accessibility",
    ["Poor", "Medium", "High"]
)

age = st.sidebar.slider(
    "Age of Property",
    0,
    30,
    5
)

current_price = st.sidebar.number_input(
    "Current Price (Lakhs)",
    10.0,
    1000.0,
    50.0
)

# ---------------- MAIN TITLE ----------------
st.title("🏡 Real Estate Investment Advisor")

st.write("### AI Powered Property Analysis")

st.divider()

# ---------------- ANALYZE BUTTON ----------------
if st.sidebar.button("Analyze Property"):

    transport_score = transport_map[
        transport.lower()
    ]

    input_data = pd.DataFrame({
        'bhk': [bhk],
        'size_in_sqft': [size],
        'nearby_schools': [schools],
        'nearby_hospitals': [hospitals],
        'transport_score': [transport_score],
        'age_of_property': [age]
    })

    # ---------------- PREDICTION ----------------
    invest_pred = clf.predict(input_data)[0]

    predicted_price = reg.predict(input_data)[0]

    future_price = predicted_price * 1.5

    profit = future_price - current_price

    # ---------------- RESULT LAYOUT ----------------
    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Investment Classification")

        if invest_pred == 1:

            st.markdown(
                """
                <div class="success-box">
                ✅ GOOD INVESTMENT
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="error-box">
                ❌ NOT RECOMMENDED
                </div>
                """,
                unsafe_allow_html=True
            )

    with col2:

        st.markdown(
            f"""
            <div class="metric-card">
            <h2>📈 Estimated Price After 5 Years</h2>
            <h1>₹ {future_price:.2f} Lakhs</h1>

            <h3>💰 Estimated Profit</h3>
            <h2>₹ {profit:.2f} Lakhs</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    # ---------------- FEATURE IMPORTANCE ----------------
    st.subheader("Feature Importance")

    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': clf.feature_importances_
    })

    importance_df = (
        importance_df
        .sort_values(
            by='Importance',
            ascending=False
        )
    )

    fig = px.bar(
        importance_df,
        x='Feature',
        y='Importance',
        title='Feature Importance'
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ---------------- CITY PRICE CHART ----------------
    st.subheader("Average Property Price by City")

    city_price = (
        df.groupby('city')['price_in_lakhs']
        .mean()
        .reset_index()
    )

    fig2 = px.bar(
        city_price,
        x='city',
        y='price_in_lakhs',
        title='Average Property Price by City'
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

st.divider()

st.caption(
    "Built using Streamlit + Machine Learning"
)
