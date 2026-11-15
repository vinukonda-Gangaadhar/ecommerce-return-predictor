import streamlit as st
import pandas as pd
import joblib


@st.cache_resource
def load_model():
    return joblib.load('ecommerce_return_model.joblib')

model = load_model()


st.title("E-Commerce Order Return Predictor")
st.write("Enter the order details below to predict if the item will be returned.")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Details")
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    is_plus_member = st.checkbox("Is Plus Member?")
    city_tier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
    region = st.selectbox("Region", ["North", "South", "East", "West"])

with col2:
    st.subheader("Order Details")
    order_value = st.number_input("Order Value (₹)", min_value=0.0, value=1500.0)
    shipping_fee = st.number_input("Shipping Fee (₹)", min_value=0.0, value=50.0)
    discount_pct = st.number_input("Discount Percentage (%)", min_value=0.0, max_value=100.0, value=10.0)
    coupon_used = st.checkbox("Coupon Used?")
    items_count = st.number_input("Number of Items", min_value=1, value=1)
    
st.subheader("Product Details")
product_avg_rating = st.slider("Product Average Rating", min_value=1.0, max_value=5.0, value=4.0)
is_size_based = st.checkbox("Is Size Based Product? (e.g., Clothing, Shoes)")
payment_method = st.selectbox("Payment Method", ["Cash on Delivery", "UPI", "Credit Card", "Debit Card", "Wallet"])


if st.button("Predict Return Status"):

    input_data = {
        'age': [age],
        'discount_pct': [discount_pct],
        'shipping_fee': [shipping_fee],
        'order_value': [order_value],
        'product_avg_rating': [product_avg_rating],
        'gender': [gender],
        'is_plus_member': [is_plus_member],
        'payment_method': [payment_method],
        'is_size_based': [is_size_based],
        'city_tier': [city_tier],
        'region': [region],
        'coupon_used': [coupon_used],
        'items_count': [items_count]
    }
    

    input_df = pd.DataFrame(input_data)
    
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] 
    
    
    if prediction == 1:
        st.error(f" High chance of Return! (Probability: {probability:.1%})")
    else:
        st.success(f"high chance not to return (Probability: {probability:.1%})")