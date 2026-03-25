import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

df = pd.read_csv("Master_Traffic_Data.csv")

df_clean = df[~df['Remarks'].str.contains('Raw Data', na=False)].copy()

numeric_cols = ['Start Time (Elapsed Sec)', 'Time to 500m (sec)', 
                'Bottleneck Delay (sec)', 'HVs Encountered', 'LVs Encountered', 'Time to 1km (sec)']

for col in numeric_cols:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# Fix Multicollinearity
df_clean['Total Vehicles'] = df_clean['HVs Encountered'] + df_clean['LVs Encountered']
df_clean = df_clean.drop(columns=['HVs Encountered', 'LVs Encountered'])

df_model = df_clean.drop(columns=['Vehicle No.', 'Actual Start Time', 'Remarks'])

# THE FIX: Keep ALL vehicle classes so we can assign positive times to all of them
df_model = pd.get_dummies(df_model, columns=['Class'], drop_first=False)

# Drop the extreme 20-minute outlier
df_model = df_model[df_model['Time to 500m (sec)'] < 400]

X = df_model.drop(columns=['Time to 1km (sec)'])
y = df_model['Time to 1km (sec)']

training_columns = X.columns
joblib.dump(training_columns, 'model_columns.pkl')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# THE FIX: Remove the intercept so the model calculates positive base times for vehicles
model = LinearRegression(fit_intercept=False)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
root_mse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n--- Model Training Complete ---")
print(f"Root Mean Squared Error: {root_mse:.2f} seconds")
print(f"R-squared Score: {r2:.4f}")

print("\nFeature Coefficients (Now showing pure positive base times for vehicles!):")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.2f}")

joblib.dump(model, 'traffic_model.pkl')