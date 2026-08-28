# CWRU Bearing Fault Diagnostics with Foundation Models

This repository contains the code and pipeline for time-series feature extraction, calibration, and optimization using CWRU vibration data.

## Project Structure

```text
ob_ttl_project/
├── data/              # Raw (.mat) and processed vibration signals
├── figures/           # Visualizations and diagnostic plots
├── models/            # Exported ONNX models and calibration artifacts
├── src/               # Core Python modules
│   ├── calibration.py
│   ├── cma_optimizer.py
│   ├── data_loader.py
│   ├── export_onnx.py
│   ├── fpm.py
│   └── run_poc.py
├── requirements.txt
└── README.md
```

## Setup & Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Jieyang02/ob_ttl_project.git
   cd ob_ttl_project
   ```

2. **Create virtual environment & install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Workflow & Usage

1. **Export Foundation Model to ONNX:**

   ```bash
   python src/export_onnx.py
   ```

2. **Run Calibration & Optimization:**
   ```bash
   python src/run_poc.py
   ```
