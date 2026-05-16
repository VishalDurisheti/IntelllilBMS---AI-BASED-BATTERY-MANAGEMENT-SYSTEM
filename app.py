import os
import time
import random
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify
from flask_cors import CORS
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# --- AI Model and Accuracy Metrics Loading ---
MODEL_FILE = 'soh_model.h5'
METRICS_FILE = 'accuracy_metrics.json'
model = None
model_performance = {"mae": "N/A", "r2_score": "N/A"}  # Default values

if os.path.exists(MODEL_FILE):
    try:
        model = load_model(MODEL_FILE)
        print("AI model loaded successfully.")
        # Load the metrics file created by the training script
        if os.path.exists(METRICS_FILE):
            with open(METRICS_FILE, 'r') as f:
                model_performance = json.load(f)
            print(f"Model performance metrics loaded: {model_performance}")
    except Exception as e:
        print(f"Error loading model or metrics: {e}")
else:
    print(f"CRITICAL: Model file '{MODEL_FILE}' not found. Please run 'generate_and_train.py' first.")

# --- Long-Term Forecast & State Initialization ---
HISTORY_FILE = 'soh_history.csv'
if not os.path.exists(HISTORY_FILE):
    print(f"Generating synthetic history file: {HISTORY_FILE}...")
    timestamps = [int((datetime.now() - timedelta(days=x)).timestamp()) for x in range(180)]
    soh_history = 100 - np.linspace(0, 3, 180) + np.random.normal(0, 0.1, 180)
    pd.DataFrame({'timestamp': sorted(timestamps), 'soh': soh_history}).to_csv(HISTORY_FILE, index=False)

history_df = pd.read_csv(HISTORY_FILE)
lr_model = LinearRegression().fit(history_df[['timestamp']], history_df['soh'])
NUM_CELLS, SEQUENCE_LENGTH = 48, 50
history_buffer = []
battery_cells = [{"id": i, "voltage": 4.1, "is_faulty": False, "is_balancing": False, "temperature": 25.5} for i in range(NUM_CELLS)]
pack_soh, fault_introduced, faulty_cell_index = 99.8, False, -1
last_fault_check_time = time.time()

# --- AI Prediction & Forecast Functions ---
def get_ai_prediction(current_features):
    if model is None: return None
    history_buffer.append(current_features)
    if len(history_buffer) > SEQUENCE_LENGTH: history_buffer.pop(0)
    if len(history_buffer) < SEQUENCE_LENGTH: return None
    
    # Simple scaling for live prediction
    live_data_scaled = (np.array(history_buffer) - [4.1, 20, 35, 90]) / [0.1, 10, 10, 10]
    reshaped_data = live_data_scaled.reshape(1, SEQUENCE_LENGTH, 4)
    predicted_soh_scaled = model.predict(reshaped_data, verbose=0)[0][0]
    return float(predicted_soh_scaled * 20 + 80) # Inverse scale

def get_long_term_forecast():
    future_ts = [int((datetime.now() + timedelta(days=x*30)).timestamp()) for x in range(1, 25)]
    future_soh = lr_model.predict(np.array(future_ts).reshape(-1, 1))
    projection = [{'x': ts, 'y': soh} for ts, soh in zip(future_ts, future_soh)]
    forecast_text = "Stable"
    for ts, soh in zip(future_ts, future_soh):
        if soh <= 80:
            forecast_text = datetime.fromtimestamp(ts).strftime("%b %Y"); break
    return {"text": forecast_text, "history": history_df.to_dict('records'), "projection": projection}

# --- Main Simulation Function ---
def simulate_battery_data():
    global pack_soh, fault_introduced, faulty_cell_index, last_fault_check_time
    for cell in battery_cells:
        if not cell["is_faulty"]: cell["voltage"] = round(max(3.0, min(4.2, cell["voltage"] + random.uniform(-0.001, 0.001))), 3)
    
    if not fault_introduced and time.time() - last_fault_check_time > 20:
        faulty_cell_index = random.randint(0, NUM_CELLS - 1)
        battery_cells[faulty_cell_index].update({"is_faulty": True, "voltage": 3.6})
        fault_introduced = True
        
    avg_voltage = sum(c['voltage'] for c in battery_cells) / NUM_CELLS
    predicted_soh = get_ai_prediction([avg_voltage, random.uniform(15, 25), 25.5, pack_soh])
    if predicted_soh: pack_soh = predicted_soh
    
    return {
        "pack_summary": {
            "total_voltage": float(round(sum(c["voltage"] for c in battery_cells), 2)),
            "avg_temperature": float(round(random.uniform(25.0, 26.0), 2)),
            "state_of_health": float(round(pack_soh, 2)),
            "alert": f"Critical Fault: Cell #{faulty_cell_index} is malfunctioning!" if fault_introduced else "None",
        },
        "cells": battery_cells,
        "long_term_forecast": get_long_term_forecast(),
        "model_performance": model_performance
    }

# --- Flask Routes ---
@app.route('/api/live-data')
def get_live_data(): return jsonify(simulate_battery_data())

@app.route('/')
def dashboard(): return render_template('index.html')

if __name__ == '__main__':
    if model:
        app.run(host='0.0.0.0', port=80, debug=False)
    else:
        print("\n--- Cannot start server: AI model is not loaded. ---")