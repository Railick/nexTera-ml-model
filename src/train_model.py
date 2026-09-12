# pyright: basic
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("data/nexterra_land_acquisition_8000.csv")

X = df.drop(columns=["case_id", "final_additional_delay_days"])

y = df["final_additional_delay_days"]

categorical_features = ["state", "district", "land_type", "acquisition_stage"]

numeric_features = [
    "case_age_days",
    "days_in_stage",
    "number_of_landowners",
    "litigation_present",
    "number_of_objections",
    "compensation_delay_days",
    "previous_stage_delay_days",
    "historical_delay_rate",
    "land_area_hectares",
]

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("numeric", "passthrough", numeric_features),
    ]
)

model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)

pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining model...")

pipeline.fit(X_train, y_train)

print("Training complete!")

y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse**0.5
r2 = r2_score(y_test, y_pred)

print("\n===== MODEL PERFORMANCE =====")

print(f"MAE  : {mae:.2f} days")
print(f"RMSE : {rmse:.2f} days")
print(f"R²   : {r2:.4f}")

results = pd.DataFrame(
    {"Actual Delay": y_test, "Predicted Delay": y_pred.round(0)}
)

print("\n===== SAMPLE PREDICTIONS =====")
print(results.head(10))

joblib.dump(pipeline, "models/nexterra_delay_model.pkl")

print("\nModel saved successfully!")
