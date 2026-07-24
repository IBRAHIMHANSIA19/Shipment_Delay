#!/usr/bin/env python
"""
Shipment Delay Prediction Script

This script loads the trained XGBoost model and uses it to predict whether a shipment
will be delayed or arrive on-time. It supports interactive single prediction, JSON file input,
and batch CSV processing.
"""

import os
import re
import sys
import json
import argparse
import joblib
import pandas as pd
import numpy as np

# Embedded label mappings matching training encoding (alphabetical unique values of cleaned dataset)
LABEL_MAPPINGS = {
    'cargo_type': {
        'Agricultural Goods': 0,
        'Automotive Parts': 1,
        'Chemicals': 2,
        'Electronics': 3,
        'Food & Beverage': 4,
        'Furniture': 5,
        'Machinery': 6,
        'Pharmaceuticals': 7,
        'Textiles': 8,
        'Toys': 9
    },
    'carrier_type': {
        'Air': 0, 
        'Rail': 1, 
        'Road': 2, 
        'Sea': 3
    },
    'category': {
        'Agricultural Goods': 0,
        'Automotive Parts': 1,
        'Chemicals': 2,
        'Electronics': 3,
        'Food & Beverage': 4,
        'Furniture': 5,
        'Machinery': 6,
        'Pharmaceuticals': 7,
        'Textiles': 8,
        'Toys': 9
    },
    'country': {
        'Australia': 0,
        'Belgium': 1,
        'Brazil': 2,
        'Canada': 3,
        'China': 4,
        'Germany': 5,
        'India': 6,
        'Indonesia': 7,
        'Japan': 8,
        'Netherlands': 9,
        'Pakistan': 10,
        'Saudi Arabia': 11,
        'Singapore': 12,
        'South Africa': 13,
        'South Korea': 14,
        'Sri Lanka': 15,
        'Thailand': 16,
        'Turkiye': 17,
        'UAE': 18,
        'UK': 19,
        'USA': 20,
        'Vietnam': 21
    },
    'customer_status': {
        'Active': 0, 
        'Inactive': 1
    },
    'customer_type': {
        'Business': 0, 
        'Individual': 1
    },
    'fuel_type': {
        'CNG': 0,
        'Diesel': 1,
        'Electric': 2,
        'Jet Fuel': 3,
        'Marine Fuel': 4,
        'Petrol': 5
    },
    'industry': {
        'Automotive': 0,
        'E-commerce': 1,
        'Electronics': 2,
        'Food Processing': 3,
        'Manufacturing': 4,
        'Pharma': 5,
        'Retail': 6,
        'Textiles': 7
    },
    'maintenance_status': {
        'Due': 0, 
        'Good': 1, 
        'Under Maintenance': 2
    },
    'priority': {
        'Express': 0, 
        'Standard': 1, 
        'Urgent': 2
    },
    'route_risk': {
        'High': 0, 
        'Low': 1, 
        'Medium': 2
    },
    'shipment_type': {
        'Export': 0, 
        'Import': 1
    },
    'shipping_mode': {
        'Air': 0, 
        'Rail': 1, 
        'Road': 2, 
        'Sea': 3
    },
    'vehicle_type': {
        'Aircraft': 0, 
        'Ship': 1, 
        'Truck': 2, 
        'Wagon': 3
    },
    'warehouse_type': {
        'Distribution': 0, 
        'Regional': 1
    },
    'weather_condition': {
        'Clear': 0,
        'Cloudy': 1,
        'Fog': 2,
        'Rain': 3,
        'Snow': 4,
        'Storm': 5
    }
}

# Sensible fallback values for features if not provided by user
FEATURE_DEFAULTS = {
    'shipping_mode': 'Sea',
    'shipment_type': 'Import',
    'priority': 'Standard',
    'weight_kg': 5000.0,
    'volume_cbm': 15.0,
    'declared_value': 10000.0,
    'insurance': False,
    'fragile_x': False,
    'carrier_type': 'Road',
    'average_rating': 4.0,
    'fleet_size': 100,
    'years_of_service': 5,
    'customer_type': 'Business',
    'industry': 'Retail',
    'country': 'USA',
    'customer_status': 'Active',
    'customs_required': False,
    'documentation_complete': True,
    'inspection_required': False,
    'cargo_type': 'Electronics',
    'category': 'Electronics',
    'hs_code': 8517.0,
    'hazardous': False,
    'perishable': False,
    'temperature_controlled': False,
    'fragile_y': False,
    'weight_per_unit': 5.0,
    'distance_km': 1000.0,
    'average_transit_days': 5,
    'route_risk': 'Low',
    'traffic_index': 30.0,
    'vehicle_type': 'Truck',
    'capacity_kg': 20000,
    'fuel_type': 'Diesel',
    'maintenance_status': 'Good',
    'vehicle_age': 4,
    'warehouse_capacity': 50000,
    'current_utilization': 75.0,
    'warehouse_type': 'Regional',
    'weather_condition': 'Clear',
    'temperature': 20.0,
    'rainfall': 0.0,
    'humidity': 50.0,
    'wind_speed': 10.0,
    'visibility': 10.0,
    'booking_month': 6,
    'booking_day': 15,
    'booking_weekday': 2,
    'ship_month': 6,
    'ship_day': 16,
    'ship_weekday': 3
}


def load_xgboost_model(model_path):
    """
    Loads XGBoost model and dynamically patches missing attributes to maintain
    compatibility across different library versions.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
        
    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Error loading model pickle with joblib: {e}")
        
    # Dynamically patch model attributes in case of xgboost version mismatch
    # (loops get_params and sets missing fields to None)
    for _ in range(50):
        try:
            model.get_params()
            break
        except AttributeError as e:
            match = re.search(r"attribute '([^']+)'", str(e))
            if match:
                attr = match.group(1)
                setattr(model, attr, None)
            else:
                raise e
                
    return model


def preprocess_shipment(raw_dict, feature_columns, label_mappings):
    """
    Preprocesses the raw shipment dictionary, converting categorical string values
    to matching label-encoded integer values, handling missing values, and formatting
    features exactly as expected by the trained model.
    """
    encoded_dict = {}
    
    for col in feature_columns:
        # Fetch user value or fall back to default
        val = raw_dict.get(col, FEATURE_DEFAULTS.get(col, 0))
        
        # 1. Handle categorical fields with label mappings
        if col in label_mappings:
            # Check if value is already a pre-encoded integer
            if isinstance(val, (int, np.integer)):
                encoded_dict[col] = int(val)
            elif isinstance(val, (float, np.floating)) and val.is_integer():
                encoded_dict[col] = int(val)
            elif isinstance(val, str) and val.isdigit() and int(val) in label_mappings[col].values():
                encoded_dict[col] = int(val)
            else:
                val_str = str(val).strip()
                # Convert direct booleans to text representation if needed
                if isinstance(val, bool):
                    val_str = str(val)
                    
                mapping = label_mappings[col]
                if val_str in mapping:
                    encoded_dict[col] = int(mapping[val_str])
                else:
                    # Default to 0 if key not found (fallback)
                    encoded_dict[col] = 0
        else:
            # 2. Handle boolean fields
            if isinstance(val, bool):
                encoded_dict[col] = 1 if val else 0
            # 3. Handle numeric fields
            else:
                try:
                    num_val = pd.to_numeric(val)
                    if pd.isna(num_val):
                        encoded_dict[col] = 0.0
                    else:
                        encoded_dict[col] = float(num_val)
                except Exception:
                    encoded_dict[col] = 0.0
                    
    # Create DataFrame with exact column order
    df = pd.DataFrame([encoded_dict])
    df = df[feature_columns]
    
    # Ensure all columns are numeric for XGBoost prediction
    for col in feature_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
    return df


def predict_delay(shipment_dict, model, feature_columns, label_mappings, threshold=0.5):
    """
    Predicts shipment delay based on the threshold.
    Returns a dictionary with prediction status, label, confidence, and raw probabilities.
    """
    try:
        df = preprocess_shipment(shipment_dict, feature_columns, label_mappings)
        
        # Predict class probabilities
        probs = model.predict_proba(df)[0]
        on_time_prob = float(round(probs[0] * 100, 2))
        delayed_prob = float(round(probs[1] * 100, 2))
        
        # Decision Rule:
        # 1. If Delayed Probability > threshold, class is "Delayed"
        # 2. Otherwise, check whichever probability is higher
        if delayed_prob > (threshold * 100):
            prediction_label = "Delayed"
            confidence = delayed_prob
        else:
            if on_time_prob >= delayed_prob:
                prediction_label = "On-Time"
                confidence = on_time_prob
            else:
                prediction_label = "Delayed"
                confidence = delayed_prob
                
        return {
            "Status": "Success",
            "Prediction": prediction_label,
            "On-Time Probability (%)": on_time_prob,
            "Delayed Probability (%)": delayed_prob,
            "Confidence (%)": confidence,
            "On-Time Probability": on_time_prob,
            "Delayed Probability": delayed_prob,
            "Confidence": confidence
        }
    except Exception as e:
        return {
            "Status": "Failed",
            "Message": str(e)
        }


def run_interactive(model, feature_columns, label_mappings):
    """
    Runs an interactive console prompt for single shipment delay prediction.
    """
    print("\n" + "=" * 50)
    print("      SHIPMENT DELAY PREDICTION INTERACTIVE MODE      ")
    print("=" * 50)
    print("Press Enter to accept default values in [brackets]\n")
    
    user_inputs = {}
    
    # Key inputs to prompt the user for (the most critical predictors)
    prompts = [
        ('shipping_mode', 'Shipping Mode (Air, Rail, Road, Sea)', 'Sea'),
        ('shipment_type', 'Shipment Type (Import, Export)', 'Import'),
        ('priority', 'Priority (Express, Standard, Urgent)', 'Standard'),
        ('weight_kg', 'Weight of shipment in kg', '5000'),
        ('distance_km', 'Transit Distance in km', '1000'),
        ('weather_condition', 'Weather Condition (Clear, Cloudy, Fog, Rain, Snow, Storm)', 'Clear'),
        ('insurance', 'Insurance coverage (True, False)', 'False'),
        ('fragile_x', 'Fragile Cargo (True, False)', 'False'),
    ]
    
    for key, desc, default in prompts:
        while True:
            val = input(f"{desc} [{default}]: ").strip()
            if not val:
                val = default
                
            # Perform basic validation for booleans
            if key in ['insurance', 'fragile_x']:
                if val.lower() in ['true', 't', '1', 'yes', 'y']:
                    user_inputs[key] = True
                    break
                elif val.lower() in ['false', 'f', '0', 'no', 'n']:
                    user_inputs[key] = False
                    break
                else:
                    print("Invalid input! Please enter True or False.")
                    continue
                    
            # For categorical strings, validate against mappings (case insensitive check)
            if key in label_mappings:
                options = label_mappings[key]
                matched_option = None
                for opt in options:
                    if opt.lower() == val.lower():
                        matched_option = opt
                        break
                if matched_option:
                    user_inputs[key] = matched_option
                    break
                else:
                    print(f"Invalid option! Valid choices are: {', '.join(options.keys())}")
                    continue
                    
            # For numeric fields, validate float conversion
            try:
                user_inputs[key] = float(val)
                break
            except ValueError:
                print("Invalid input! Please enter a numeric value.")
                
    # Fill remaining columns with defaults
    for col in feature_columns:
        if col not in user_inputs:
            user_inputs[col] = FEATURE_DEFAULTS.get(col, 0)
            
    print("\nProcessing prediction...")
    res = predict_delay(user_inputs, model, feature_columns, label_mappings)
    
    if res["Status"] == "Success":
        print("\n" + "*" * 50)
        print(f"  PREDICTION: {res['Prediction'].upper()}")
        print(f"  Confidence: {res['Confidence (%)']}%")
        print("-" * 50)
        print(f"  On-Time Probability: {res['On-Time Probability (%)']}%")
        print(f"  Delayed Probability: {res['Delayed Probability (%)']}%")
        print("*" * 50 + "\n")
    else:
        print(f"\nPrediction failed: {res['Message']}\n")


def run_batch_csv(csv_path, output_path, model, feature_columns, label_mappings, threshold=0.5):
    """
    Runs batch predictions on a CSV file and saves results to an output CSV file.
    """
    print(f"Loading input CSV: {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return
        
    print("Processing batch predictions...")
    predictions = []
    on_time_probs = []
    delayed_probs = []
    confidences = []
    
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        res = predict_delay(row_dict, model, feature_columns, label_mappings, threshold)
        if res["Status"] == "Success":
            predictions.append(res["Prediction"])
            on_time_probs.append(res["On-Time Probability (%)"])
            delayed_probs.append(res["Delayed Probability (%)"])
            confidences.append(res["Confidence (%)"])
        else:
            predictions.append("ERROR")
            on_time_probs.append(0.0)
            delayed_probs.append(0.0)
            confidences.append(0.0)
            
    # Add new output columns to the DataFrame
    df["predicted_status"] = predictions
    df["on_time_probability_pct"] = on_time_probs
    df["delayed_probability_pct"] = delayed_probs
    df["prediction_confidence_pct"] = confidences
    
    try:
        df.to_csv(output_path, index=False)
        print(f"Successfully saved batch predictions to: {output_path}")
    except Exception as e:
        print(f"Error saving output CSV file: {e}")


def main():
    parser = argparse.ArgumentParser(description="Predict shipment delays using the best trained XGBoost model.")
    parser.add_argument("--model", type=str, default="best_xgboost_model.pkl", help="Path to best_xgboost_model.pkl")
    parser.add_argument("--features", type=str, default="feature_columns.pkl", help="Path to feature_columns.pkl")
    parser.add_argument("--csv", type=str, help="Path to input CSV file for batch prediction")
    parser.add_argument("--output", type=str, default="predictions_output.csv", help="Path to save prediction output CSV")
    parser.add_argument("--json", type=str, help="Path to input JSON file representing a single shipment dict")
    parser.add_argument("--threshold", type=float, default=0.5, help="Delay probability threshold (default: 0.5)")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive single prediction mode")
    
    args = parser.parse_args()
    
    print("Initializing Prediction Pipeline...")
    
    # 1. Load feature columns list
    try:
        feature_columns = joblib.load(args.features)
        print(f"Loaded {len(feature_columns)} feature columns from {args.features}")
    except Exception as e:
        print(f"Error loading feature columns list: {e}")
        sys.exit(1)
        
    # 2. Load and patch XGBoost model
    try:
        model = load_xgboost_model(args.model)
        print(f"Loaded and patched XGBoost model from {args.model}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
        
    # 3. Choose running mode
    if args.csv:
        run_batch_csv(args.csv, args.output, model, feature_columns, LABEL_MAPPINGS, args.threshold)
    elif args.json:
        try:
            with open(args.json, 'r') as f:
                shipment_dict = json.load(f)
            res = predict_delay(shipment_dict, model, feature_columns, LABEL_MAPPINGS, args.threshold)
            print(json.dumps(res, indent=4))
        except Exception as e:
            print(f"Error processing JSON file: {e}")
    elif args.interactive or len(sys.argv) == 1:
        run_interactive(model, feature_columns, LABEL_MAPPINGS)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
