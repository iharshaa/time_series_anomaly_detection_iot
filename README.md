# Time Series Anomaly Detection for IoT Sensors

## Project Overview

This project implements a complete, end-to-end anomaly detection solution for multivariate time-series IoT sensor data. The solution uses real-world AWS CloudWatch metrics from the Numenta Anomaly Benchmark (NAB) dataset and demonstrates both classical machine learning and deep learning approaches for anomaly detection.

## Dataset

**Source**: [Numenta Anomaly Benchmark (NAB)](https://github.com/numenta/NAB)

The NAB dataset contains over 50 labeled real-world and artificial time-series data files designed for evaluating anomaly detection algorithms in streaming, real-time applications. This project focuses on AWS CloudWatch metrics including CPU utilization and network traffic.

## Models Implemented

1. **Isolation Forest** (Classical/Unsupervised)
   - Robust anomaly detection using random forest partitioning
   - Fast training and inference
   - No assumptions about data distribution

2. **LSTM Autoencoder** (Deep Learning)
   - Captures temporal dependencies in time-series data
   - Learns normal behavior patterns
   - Uses reconstruction error for anomaly scoring

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. Clone or download this repository
2. Navigate to the project directory:
   ```bash
   cd time-series-anomaly-detection-iot
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Jupyter Notebook

1. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Open `anomaly_detection_notebook.ipynb` in the browser

3. Run all cells sequentially (Cell → Run All) or execute cells one by one

### Expected Runtime
- Data loading and preparation: ~2-3 minutes
- Feature engineering: ~1-2 minutes
- Isolation Forest training: ~30 seconds
- LSTM Autoencoder training: ~5-10 minutes (CPU) or ~2-3 minutes (GPU)
- Total runtime: ~15-20 minutes

## Project Structure

```
time-series-anomaly-detection-iot/
│
├── anomaly_detection_notebook.ipynb  # Main deliverable - complete analysis
├── data_loader.py                     # NAB dataset loading utilities
├── requirements.txt                   # Project dependencies
├── README.md                          # This file
│
└── data/                              # Downloaded NAB data (auto-created)
    └── NAB/
        ├── data/
        └── labels/
```

## Key Features

- **Comprehensive EDA**: Statistical analysis and visualizations
- **Feature Engineering**: Rolling statistics, lag features, rate of change
- **Dual Approach**: Classical ML and Deep Learning comparison
- **Evaluation Metrics**: Precision, Recall, F1-Score with ground truth labels
- **Business Insights**: Actionable recommendations for manufacturing/IoT scenarios
- **Production-Ready**: Modular, well-documented, reproducible code

## Results

The notebook includes:
- Detailed performance comparison between Isolation Forest and LSTM Autoencoder
- Visual anomaly detection results overlaid on time-series data
- Comprehensive evaluation metrics with confusion matrices
- Business insights and operational recommendations

## Limitations & Future Work

- Dataset limited to AWS CloudWatch metrics (single domain)
- Models may need retraining for different IoT sensor types
- Future enhancements: Transformer models, ensemble methods, real-time streaming architecture



## License

This project uses the NAB dataset which is available under Apache 2.0 License.

---

For questions or issues, please refer to the detailed documentation within the Jupyter notebook.
