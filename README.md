# Credit Card Default Predictor
This project implements a predictive pipeline to identify credit card defaults using the CC Default dataset. It benchmarks classical ML (XGBoost, Random Forest) against deep learning architectures (LSTM, Transformer) built in PyTorch. The repository features robust EDA, automated sequence preprocessing, and stratified K-fold cross-validation.

---
## Project Structure

This repository uses a flat structure for transparency and ease of execution. All core logic and execution scripts are located in the root directory:

├── cc_eda_00.py        # Exploratory Data Analysis & Visualizations
├── cc_pipeline_01.py   # Main pipeline: benchmarks classical vs. deep models
├── cc_timeseries_02.py # Model architectures (LSTM/Transformer) & K-Fold logic
├── requirements.txt    # Project dependencies
├── LICENSE             # MIT License
├── .gitignore          # Data and environment exclusions
└── README.md           # Project documentation

## Key Features
Deep Learning Architectures: Custom-built LSTM and Transformer models designed to handle temporal billing and payment sequences.
Hybrid Data Processing: Dual-stream preprocessing that handles static demographic data and sequential financial data simultaneously.
Robust Validation: Utilizes Stratified K-Fold Cross-Validation and Random Oversampling to address class imbalance and ensure model stability.
Benchmarking: Comparative analysis against classical classifiers including XGBoost, Random Forest, and Naive Bayes.

