from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

model = joblib.load('ecommerce_return_model.joblib')

app = FastAPI(title="Return Prediction API")
class OrderData(BaseModel):
    age: int
    discount_pct: float
    shipping_fee: float
    order_value: float
    product_avg_rating: float
    gender: str
    is_plus_member: bool
    payment_method: str
    is_size_based: bool
    city_tier: str
    region: str
    coupon_used: bool
    items_count: int

@app.get("/")
def read_root():
    return {"message": "Welcome to the E-commerce Return Predictor API"}

@app.post("/predict")
def predict_return(data: OrderData):
    input_dict = data.dict()
    input_df = pd.DataFrame([input_dict])
    
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    result = "Returned" if prediction == 1 else "Not Returned"
    
    return {
        "prediction": result,
        "return_probability": round(float(probability), 2)
    }