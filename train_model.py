"""
train_model.py
Run this once to generate model.pkl used by app.py

Usage:
    python train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')

# ── Load & clean ────────────────────────────────────────────────────────────────
df = pd.read_csv('Car Dataset Processed.csv')

# Remove extreme outliers (data entry errors)
df = df[df['price(in lakhs)'] < 200]
df = df[df['torque(Nm)'] < 50000]
print(f"Dataset size after cleaning: {len(df)} rows")

# ── Encode categoricals ─────────────────────────────────────────────────────────
d1 = {'Comprehensive':0,'Third Party insurance':1,'Third Party':1,'Zero Dep':2,'Not Available':3}
d2 = {'Petrol':0,'Diesel':1,'CNG':2}
d3 = {'First Owner':1,'Second Owner':2,'Third Owner':3,'Forth Owner':4,'Fifth Owner':5}
d4 = {'Manual':0,'Automatic':1}

df['insurance_validity'] = df['insurance_validity'].map(d1)
df['fuel_type']          = df['fuel_type'].map(d2)
df['ownsership']         = df['ownsership'].map(d3)
df['transmission']       = df['transmission'].map(d4)

# ── Features ────────────────────────────────────────────────────────────────────
FEATURES = [
    'insurance_validity', 'fuel_type', 'kms_driven', 'ownsership',
    'transmission', 'seats', 'manufacturing_year', 'mileage(kmpl)',
    'engine(cc)', 'max_power(bhp)'
]

X = df[FEATURES]
Y = df['price(in lakhs)']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# ── Train XGBoost ───────────────────────────────────────────────────────────────
model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)

model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          verbose=False)

# ── Evaluate ────────────────────────────────────────────────────────────────────
r2  = model.score(X_test, y_test)
mae = mean_absolute_error(y_test, model.predict(X_test))
print(f"Test R²  : {r2:.4f}")
print(f"Test MAE : ₹{mae:.2f} Lakhs")

# ── Save ────────────────────────────────────────────────────────────────────────
pickle.dump(model, open('model.pkl', 'wb'))
print("✅  model.pkl saved successfully!")
