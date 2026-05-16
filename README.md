# IntelliBMS: AI-Based Battery Management System

IntelliBMS is an intelligent, data-driven Battery Management System backend designed to monitor, analyze, and forecast the health of lithium-ion battery packs using machine learning. 

Traditional BMS architectures rely on rigid, threshold-based alerts that only flag hardware issues after a failure has occurred. IntelliBMS introduces a predictive approach, tracking non-linear telemetry trends to catch cell degradation early and forecast exact battery life progression.

---

## 🤖 How It Works

The project is split into two core computational pipelines:

### 1. Deep Learning Training Pipeline (`generate_and_train.py`)
* **Synthetic Telemetry Generation:** Simulates a dataset of 10,000 operational data points tracking voltage, current, and temperature alongside realistic battery degradation curves.
* **Time-Series LSTM Network:** Implements a Recurrent Neural Network (RNN) using **LSTM layers** built with TensorFlow. It processes multi-dimensional historical sequences (50 time-steps) to output a highly precise State of Health (SoH) percentage.
* **Rigorous Validation:** Evaluates the trained network on an unseen test set to compute the Mean Absolute Error (MAE) and R-squared (R²) scoring before deploying the model binary (`soh_model.h5`).

### 2. Real-Time Simulation API (`app.py`)
* **Live Inference Engine:** Loads the serialized LSTM model to run live predictions based on continuous runtime pack telemetry.
* **48-Cell Pack Matrix:** Simulates a physical multi-cell configuration, keeping track of individual cell voltages, average temperatures, and localized anomalies.
* **Automated Fault Injection:** Features an automated background routine that introduces random cell voltage drops to test the responsiveness of downstream safety flags and system alerts.
* **Long-Term Prognostics:** Leverages a Scikit-learn **Linear Regression** model to map historical degradation data and project the exact month the pack's capacity is expected to drop below the critical 80% threshold.

---
<img width="1440" height="900" alt="dashboard" src="https://github.com/user-attachments/assets/59ff8ccb-ccef-4921-a074-661497f6d3c3" />


## 🛠️ Tech Stack & Skills Demonstrated

* **Frameworks & APIs:** Flask, Flask-CORS
* **Machine Learning & Deep Learning:** TensorFlow, Keras, Scikit-learn
* **Data Engineering & Analytics:** NumPy, Pandas
* **Core Competencies:** Time-series sequence forecasting, full-cycle ML pipeline development (data engineering to production deployment), and hardware logic simulation.
