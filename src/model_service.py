# pyright: basic
import joblib
import numpy as np
import pandas as pd
import shap

# Load trained model

pipeline = joblib.load("models/nexterra_delay_model.pkl")

preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]

# SHAP explainer

explainer = shap.TreeExplainer(model)

# Feature explanations

feature_info = {
    "numeric__number_of_landowners": (
        "Number of landowners",
        "A large number of landowners can increase coordination and settlement complexity.",
    ),
    "numeric__days_in_stage": (
        "Days in current stage",
        "A case that has already spent significant time in its current stage may face further delay.",
    ),
    "numeric__compensation_delay_days": (
        "Compensation delay",
        "Existing compensation delays can indicate difficulty completing the acquisition process.",
    ),
    "numeric__historical_delay_rate": (
        "Historical delay rate",
        "A higher historical delay rate indicates that similar cases have frequently experienced delays.",
    ),
    "numeric__litigation_present": (
        "Litigation present",
        "Litigation can introduce legal uncertainty and extend the acquisition timeline.",
    ),
    "numeric__number_of_objections": (
        "Number of objections",
        "More objections can increase the time required for review and resolution.",
    ),
    "numeric__previous_stage_delay_days": (
        "Previous stage delay",
        "Delays in earlier stages can indicate unresolved issues affecting the current case.",
    ),
    "numeric__land_area_hectares": (
        "Land area",
        "Larger acquisition areas can involve greater coordination and processing requirements.",
    ),
    "numeric__case_age_days": (
        "Case age",
        "Older cases may indicate that the acquisition process has already encountered difficulties.",
    ),
}

# Prediction function


def predict_case(case_data):

    # Convert incoming data to DataFrame
    case = pd.DataFrame([case_data])

    # Predict delay

    predicted_delay = pipeline.predict(case)[0]

    predicted_delay = round(float(predicted_delay))

    # Risk level

    if predicted_delay < 120:
        risk = "LOW"

    elif predicted_delay < 200:
        risk = "MEDIUM"

    else:
        risk = "HIGH"

    # Transform case for SHAP

    transformed_case = preprocessor.transform(case)

    if hasattr(transformed_case, "toarray"):
        transformed_case = transformed_case.toarray()

    transformed_case = np.asarray(transformed_case, dtype=np.float64)

    # Calculate SHAP values

    shap_values = explainer.shap_values(transformed_case)

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 2:
        shap_values = shap_values[0]

    # Feature names
    feature_names = preprocessor.get_feature_names_out()

    # Build explanation

    explanation = pd.DataFrame({"feature": feature_names, "shap_value": shap_values})

    # Only factors increasing delay
    explanation = explanation[explanation["shap_value"] > 0]

    explanation = explanation.sort(by="shap_value", ascending=False)

    # Convert to clean reasons

    reasons = []

    for _, row in explanation.iterrows():
        feature = str(row["feature"])
        impact = float(row["shap_value"])

        # Normal numeric feature
        if feature in feature_info:
            name, reason = feature_info[feature]

            value = case_data[feature.replace("numeric__", "")]

            if feature == "numeric__litigation_present":
                value = "Yes" if value == 1 else "No"

            elif feature == "numeric__historical_delay_rate":
                value = f"{value * 100:.0f}%"

            reasons.append(
                {
                    "feature": name,
                    "value": value,
                    "impact_days": round(impact, 2),
                    "reason": reason,
                }
            )

        # Acquisition stage
        elif feature.startswith("categorical__acquisition_stage_"):
            stage = feature.replace("categorical__acquisition_stage_", "")

            reasons.append(
                {
                    "feature": "Acquisition stage",
                    "value": stage,
                    "impact_days": round(impact, 2),
                    "reason": (
                        f"The case is currently in the "
                        f"{stage} stage, which contributes "
                        "to the predicted delay."
                    ),
                }
            )

        # State
        elif feature.startswith("categorical__state_"):
            state = feature.replace("categorical__state_", "")

            reasons.append(
                {
                    "feature": "State",
                    "value": state,
                    "impact_days": round(impact, 2),
                    "reason": (
                        f"Historical patterns learned for "
                        f"{state} contribute to the prediction."
                    ),
                }
            )

    # Keep top 5 reasons
    reasons = reasons[:5]

    # Return result

    return {
        "predicted_additional_delay_days": predicted_delay,
        "risk_level": risk,
        "reasons": reasons,
    }
