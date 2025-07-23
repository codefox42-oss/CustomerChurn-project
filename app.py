# app.py
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\load_data.csv.csv")

st.title("📉 Customer Churn Prediction")

# Preprocess data
df = df.dropna()
le = LabelEncoder()
for column in df.select_dtypes(include='object'):
    df[column] = le.fit_transform(df[column])

# Split features and target
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Get list of features
features = list(X.columns)

# Ask for user input dynamically (simplified for demo with only tenure)
st.subheader("Input Customer Data")
tenure = st.number_input("Tenure (in months):", min_value=0, max_value=100, value=12)

# Create a sample input based on a random test row, and overwrite 'tenure'
sample = X_test.iloc[0].copy()
sample['tenure'] = tenure

# Predict churn probability
churn_prob = model.predict_proba([sample])[0][1]
st.write(f"🔮 **Predicted Churn Probability:** {churn_prob:.2f}")
