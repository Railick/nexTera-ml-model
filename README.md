# NexTerra ML Model

AI-powered predictive analytics system for early detection of land acquisition delays.

NexTerra predicts how many additional days a land acquisition case may be delayed and explains the main factors contributing to that prediction.

---

## What NexTerra Does

The ML system takes information about a land acquisition case, such as:

- State and district
- Type of land
- Current acquisition stage
- Case age
- Number of days spent in the current stage
- Number of landowners
- Litigation status
- Number of objections
- Compensation delays
- Previous-stage delays
- Historical delay rate
- Land area

It then produces:

1. Expected additional delay in days
2. Risk level
3. Main factors contributing to the prediction

Example:

```text
Predicted additional delay: 214 days
Risk level: HIGH

Main reasons:
- Previous stage delay: +22.68 days
- Compensation delay: +6.87 days
- Number of objections: +3.64 days
- Number of landowners: +1.34 days
- Historical delay rate: +1.11 days
