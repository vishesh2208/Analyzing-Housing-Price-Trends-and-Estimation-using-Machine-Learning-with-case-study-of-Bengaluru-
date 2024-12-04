import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import joblib
import tempfile

@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

data = load_data("Bengaluru_House_Data_Cleaned (1).csv")

model = joblib.load('linear_regression_model (1).pkl')

st.title("Analyzing Housing Price Trends and Estimation using Machine Learning with case study of Bengaluru")

st.write("### Dataset Overview")
st.dataframe(data.head())

st.sidebar.title("Filters")
location = st.sidebar.multiselect("Select Locations", data['location'].unique())
bhk_filter = st.sidebar.slider("Select Number of BHKs", int(data['bhk'].min()), int(data['bhk'].max()), (1, 4))
price_range = st.sidebar.slider("Price Range (in lakhs)", int(data['price'].min()), int(data['price'].max()), (50, 200))

filtered_data = data[
    (data['location'].isin(location) if location else True) &
    (data['bhk'].between(bhk_filter[0], bhk_filter[1])) &
    (data['price'].between(price_range[0], price_range[1]))
]

st.write("### Filtered Data")
st.dataframe(filtered_data)

st.write("### Price Distribution by Location")
top_locations = data['location'].value_counts().head(10).index
price_data = data[data['location'].isin(top_locations)]
fig = px.box(price_data, x='location', y='price', color='location', title="Price Distribution by Location")
st.plotly_chart(fig)

st.write("### Scatter Plot: Total Sqft vs. Price")
fig = px.scatter(filtered_data, x='total_sqft', y='price', color='location', title="Area vs Price")
st.plotly_chart(fig)

st.write("### Correlation Heatmap")
corr_data = data[['total_sqft', 'price', 'bath', 'balcony', 'bhk', 'price_per_sqft']].dropna()
corr_matrix = corr_data.corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

st.write("### Area Type Distribution")
area_data = data.groupby('area_type')['price'].sum().reset_index()
fig = px.treemap(area_data, path=['area_type'], values='price', title="Area Type Contribution")
st.plotly_chart(fig)

st.write("### Property Map")
map_data = data[['location', 'price']].dropna()
map_data['latitude'] = np.random.uniform(12.9, 13.1, len(map_data))  # Mock coordinates
map_data['longitude'] = np.random.uniform(77.5, 77.7, len(map_data))
fig = px.scatter_mapbox(
    map_data,
    lat='latitude',
    lon='longitude',
    color='price',
    size='price',
    hover_name='location',
    mapbox_style="open-street-map",
    title="Property Prices in Bengaluru"
)
st.plotly_chart(fig)

st.write("### Pair Plot for Numerical Features")
pairplot_data = data[['total_sqft', 'price', 'bath', 'balcony', 'bhk']].dropna()
with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
    pairplot_fig = sns.pairplot(pairplot_data)
    pairplot_fig.savefig(temp_file.name)
    st.image(temp_file.name)

st.write("### 3D Scatter Plot: Price vs Total Sqft vs Number of Bathrooms")
fig = px.scatter_3d(
    filtered_data,
    x='total_sqft',
    y='price',
    z='bath',
    color='location',
    title="3D Scatter Plot: Price vs Total Sqft vs Bathrooms",
    labels={'bath': 'Number of Bathrooms'}
)
st.plotly_chart(fig)

st.write("### Dependency of Price on Features")
fig = px.scatter_3d(data, x='total_sqft', y='bath', z='price', color='is_luxury', title="3D Dependency of Price")
st.plotly_chart(fig)

st.write("### Predict Property Price")

total_sqft = st.number_input("Enter Total Sqft", min_value=0, value=1000)
bhk = st.slider("Select Number of BHKs", int(data['bhk'].min()), int(data['bhk'].max()), 2)
bath = st.slider("Select Number of Bathrooms", int(data['bath'].min()), int(data['bath'].max()), 2)
balcony = st.slider("Select Number of Balconies", int(data['balcony'].min()), int(data['balcony'].max()), 1)

price_per_sqft = total_sqft * 500  
input_features = np.array([[total_sqft, bhk, bath, balcony, price_per_sqft]])
predicted_price = model.predict(input_features)
st.write(f"Predicted Price: ₹ {predicted_price[0]:,.2f} Lakhs")

st.markdown(
    """
    <hr style='border-top: 1px solid #ddd; margin-top: 30px; margin-bottom: 10px;'/>
    <p style='text-align: center; color: grey; font-size: 0.9em;'>© 2024 Developed by Mukul and Vishesh</p>
    """, unsafe_allow_html=True
)












