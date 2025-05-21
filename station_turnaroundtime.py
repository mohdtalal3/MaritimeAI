import joblib
import pandas as pd

# Load models
model_turnaround = joblib.load("model_turnaround_simple.pkl")
model_stationing = joblib.load("model_stationing_simple.pkl")

def predict_simple(ship_type, source_port, destination_port):
    input_df = pd.DataFrame([{
        "Ship_Type": ship_type,
        "First_Port": source_port,
        "Last_Port": destination_port
    }])
    pred_turnaround = model_turnaround.predict(input_df)[0]
    pred_stationing = model_stationing.predict(input_df)[0]
    return {
        "Predicted_Overall_Turnaround_Time_Days": round(pred_turnaround, 2),
        "Predicted_Stationing_Days": max(1, round(pred_stationing))
    }

# Example usage
result = predict_simple("Cargo", "Aalborg", "Havdrup")
print(result)
