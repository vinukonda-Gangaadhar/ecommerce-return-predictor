import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("india_ecommerce_orders_returns.csv", low_memory=False)


selected_cols = [
    'is_returned', 'age', 'discount_pct', 'shipping_fee', 'order_value', 
    'product_avg_rating', 'gender', 'is_plus_member', 'payment_method', 
    'is_size_based', 'city_tier', 'region', 'coupon_used', 'items_count'
]
df = df[selected_cols].copy()

df_majority = df[df.is_returned == False]
df_minority = df[df.is_returned == True]

df_majority_downsampled = resample(df_majority, 
                                 replace=False,    
                                 n_samples=len(df_minority),     
                                 random_state=42) 

df_balanced = pd.concat([df_majority_downsampled, df_minority])

X = df_balanced.drop('is_returned', axis=1)
y = df_balanced['is_returned'].astype(int) # Convert boolean to 1 and 0

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

clf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf_pipeline.fit(X_train, y_train)
y_pred = clf_pipeline.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
'''
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
'''
import joblib
joblib.dump(clf_pipeline, 'ecommerce_return_model.joblib')