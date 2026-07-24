#!/usr/bin/env python
"""
Script to test the prediction pipeline on 10 random shipments from the dataset
and print the results in the terminal.
"""

import pandas as pd
import predict

def main():
    model_path = "best_xgboost_model.pkl"
    features_path = "feature_columns.pkl"
    csv_path = "cleaned_merged_dataset_v3 (1).csv"
    
    print("=" * 70)
    print("  LOADING PREDICTION MODEL AND DATASET")
    print("=" * 70)
    
    # 1. Load feature columns list
    try:
        feature_columns = predict.joblib.load(features_path)
    except Exception as e:
        print(f"Error loading feature columns list: {e}")
        return
        
    # 2. Load model
    try:
        model = predict.load_xgboost_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    # 3. Load dataset and sample 10 random rows
    try:
        df = pd.read_csv(csv_path)
        sample_df = df.sample(n=10, random_state=42)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print("\n" + "=" * 70)
    print("  RUNNING DELAY PREDICTIONS ON 10 RANDOM SHIPMENTS")
    print("=" * 70)
    
    results = []
    for idx, row in sample_df.iterrows():
        row_dict = row.to_dict()
        res = predict.predict_delay(row_dict, model, feature_columns, predict.LABEL_MAPPINGS)
        
        if res["Status"] == "Success":
            results.append({
                "Shipment ID": row_dict.get("shipment_id", "N/A")[:8] + "...",
                "Mode": row_dict.get("shipping_mode", "N/A"),
                "Priority": row_dict.get("priority", "N/A"),
                "Distance": f"{row_dict.get('distance_km', 0):.1f} km",
                "Prediction": res["Prediction"],
                "Confidence": f"{res['Confidence (%)']:.2f}%"
            })
            
    # Format and print the table to the terminal
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
