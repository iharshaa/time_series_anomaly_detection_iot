# Time Series Anomaly Detection for IoT Sensors - Project Summary

## Executive Overview

This project implements a complete, end-to-end **anomaly detection solution** for multivariate time-series IoT sensor data. The implementation demonstrates both classical machine learning and deep learning approaches using real-world AWS CloudWatch metrics from the Numenta Anomaly Benchmark (NAB) dataset.

**Project Status**: ✅ Complete and Operational  
**Date**: December 2025

---

## 🎯 Project Objectives

### Primary Goal
Develop a production-ready anomaly detection system for IoT sensors in manufacturing environments to:
- Detect equipment failures before they occur
- Reduce downtime costs (est. $5K-$50K per hour)
- Improve worker safety
- Enable predictive maintenance strategies

### Key Deliverables
1. ✅ Fully functional Jupyter notebook with complete analysis
2. ✅ Custom data loader for NAB dataset
3. ✅ Implementation of Isolation Forest (classical ML)
4. ✅ Implementation of LSTM Autoencoder (deep learning)
5. ✅ Comprehensive evaluation metrics and visualizations
6. ✅ Business insights and operational recommendations

---

## 📊 Dataset Information

**Source**: [Numenta Anomaly Benchmark (NAB)](https://github.com/numenta/NAB)

### Dataset Characteristics
- **Type**: Real-world AWS CloudWatch metrics (labeled)
- **Metrics Included**: 
  - EC2 CPU Utilization (multiple instances)
  - Network Traffic (in/out)
  - Request Counts
- **Total Records**: 12,096 data points
- **Time Span**: 60 days (Feb 14 - Apr 16, 2014)
- **Anomaly Rate**: 23.22% (2,809 anomalous points)
- **Data Quality**: ✅ No missing values, no duplicates

### Why NAB?
NAB is an industry-standard benchmark specifically designed for evaluating anomaly detection algorithms in streaming, real-time applications - making it ideal for IoT sensor monitoring scenarios.

---

## 🔬 Technical Approach

### Models Implemented

#### 1. **Isolation Forest** (Classical/Unsupervised)
- **Algorithm Type**: Ensemble method using random forest partitioning
- **Training Time**: ~30 seconds
- **Advantages**:
  - Fast training and inference
  - No assumptions about data distribution
  - Robust to outliers
  - Excellent for high-dimensional data

#### 2. **LSTM Autoencoder** (Deep Learning)
- **Architecture**: Encoder-Decoder neural network
- **Training Time**: 5-10 minutes (CPU) / 2-3 minutes (GPU)
- **Advantages**:
  - Captures temporal dependencies
  - Learns normal behavior patterns automatically
  - Uses reconstruction error for anomaly scoring
  - Better for sequential data patterns

### Feature Engineering
The project includes sophisticated feature engineering:
- **Rolling Statistics**: Moving averages, standard deviations
- **Lag Features**: Previous time step values
- **Rate of Change**: Temporal derivatives
- **Time-based Features**: Hour of day, day of week patterns

---

## 💻 Technical Stack

### Dependencies
```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
tensorflow>=2.20.0
matplotlib>=3.6.0
seaborn>=0.12.0
jupyter>=1.0.0
requests>=2.28.0
```

### Project Structure
```
time-series-anomaly-detection-iot/
│
├── anomaly_detection_notebook.ipynb  # Main analysis (574KB, comprehensive)
├── data_loader.py                     # Custom NAB dataset loader (9.8KB)
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
│
└── data/                              # Downloaded NAB data
    └── NAB/
        ├── data/         # 50+ time-series CSV files
        └── labels/       # Ground truth anomaly labels
```

---

## 📈 Results & Performance

### Model Comparison

| Metric | Isolation Forest | LSTM Autoencoder |
|--------|-----------------|------------------|
| Precision | High | Very High |
| Recall | Moderate | High |
| F1-Score | Good | Excellent |
| Training Time | ~30 sec | 5-10 min |
| Inference Speed | Fast | Moderate |
| Interpretability | Good | Limited |

### Key Findings
1. **Isolation Forest**: Best for rapid deployment and explainability
2. **LSTM Autoencoder**: Superior for complex temporal patterns
3. **Ensemble Approach**: Combining both models recommended for production

### Visualizations
The notebook includes:
- Time-series plots with anomaly highlighting
- Confusion matrices for both models
- ROC curves and precision-recall curves
- Feature importance analysis
- Reconstruction error distributions

---

## 💼 Business Impact

### Manufacturing IoT Applications

#### Predictive Maintenance Benefits
- ✅ **30-50% reduction** in downtime
- ✅ **25-30% lower** maintenance costs
- ✅ **20-40% extended** equipment lifespan
- ✅ **Improved safety** preventing catastrophic failures
- ✅ **Data-driven** operational decisions

#### Real-World Use Cases
1. **Equipment Monitoring**: Detect bearing wear, motor degradation
2. **Energy Management**: Identify unusual consumption patterns
3. **Quality Control**: Spot production anomalies in real-time
4. **Security**: Detect unauthorized access or cyberattacks
5. **Environmental Monitoring**: Track pollution, temperature variations

---

## 🚀 Usage Instructions

### Quick Start

```bash
# 1. Navigate to project directory
cd time-series-anomaly-detection-iot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download NAB dataset (automatic on first run)
python data_loader.py

# 4. Launch Jupyter Notebook
jupyter notebook

# 5. Open and run anomaly_detection_notebook.ipynb
```

### Expected Runtime
- Data loading: 2-3 minutes
- Feature engineering: 1-2 minutes
- Isolation Forest: ~30 seconds
- LSTM Autoencoder: 5-10 minutes (CPU)
- **Total**: 15-20 minutes

---

## 🔍 Key Features

### Data Loader (`data_loader.py`)
- Automatic NAB dataset download from GitHub
- Multivariate dataset creation from multiple CSV files
- Anomaly label loading and alignment
- Time-series preprocessing and validation
- 294 lines of production-ready code

### Jupyter Notebook (`anomaly_detection_notebook.ipynb`)
Comprehensive 9-section analysis:
1. **Problem Understanding**: Business context and anomaly types
2. **Data Preparation & EDA**: Statistical analysis and quality checks
3. **Feature Engineering**: Advanced time-series features
4. **Model Implementation**: Both Isolation Forest and LSTM
5. **Evaluation**: Detailed performance metrics
6. **Visualization**: Professional plots and charts
7. **Business Insights**: Actionable recommendations
8. **Limitations**: Honest assessment of constraints
9. **Documentation**: Complete README generation

---

## ⚠️ Limitations & Future Work

### Current Limitations
- Dataset limited to AWS CloudWatch metrics (single domain)
- Models may require retraining for different sensor types
- Real-time streaming architecture not implemented
- Limited to 4 metrics (CPU and network)

### Proposed Enhancements
1. **Advanced Models**: Transformer-based architectures, VAE
2. **Ensemble Methods**: Voting classifiers, stacking
3. **Real-time Processing**: Apache Kafka/Spark integration
4. **Cloud Deployment**: Containerization (Docker), Kubernetes
5. **MLOps**: Model versioning, monitoring, A/B testing
6. **Multi-domain**: Testing on diverse IoT sensor types

---

## 📚 Technical Highlights

### Code Quality
- ✅ Modular, well-documented code
- ✅ Object-oriented design for data loader
- ✅ Type hints and docstrings
- ✅ Error handling and validation
- ✅ Reproducible results (random seeds set)

### Best Practices
- Clean separation of concerns
- Efficient data loading with streaming
- Memory-conscious operations
- Comprehensive logging
- Production-ready error messages

---

## 🎓 Learning Outcomes

This project demonstrates expertise in:
1. **Machine Learning**: Classical and deep learning algorithms
2. **Time-Series Analysis**: Feature engineering, temporal patterns
3. **Data Engineering**: ETL pipelines, data quality
4. **Software Engineering**: Clean code, documentation
5. **Business Acumen**: Translating technical results to business value
6. **Research**: Understanding and implementing academic benchmarks

---

## 📝 License & Attribution

- **NAB Dataset**: Available under Apache 2.0 License
- **Project Code**: Educational/Portfolio use
- **Citations**: Numenta Anomaly Benchmark (2015)

---

## 🔗 References

1. Numenta Anomaly Benchmark: https://github.com/numenta/NAB
2. Isolation Forest Paper: Liu et al. (2008)
3. LSTM Paper: Hochreiter & Schmidhuber (1997)
4. Autoencoder Anomaly Detection: Various research papers

---

## 📞 Contact & Support

For questions, issues, or collaboration:
- Review detailed documentation in `anomaly_detection_notebook.ipynb`
- Check `README.md` for setup instructions
- Examine `data_loader.py` for data pipeline details

---

**Document Generated**: December 2025  
**Last Updated**: After successful project completion  
**Version**: 1.0

---

## 🎯 Quick Reference

| Component | Status | Purpose |
|-----------|--------|---------|
| Dataset Download | ✅ Automated | NAB data acquisition |
| Data Loading | ✅ Complete | Multivariate time-series |
| EDA | ✅ Comprehensive | Statistical analysis |
| Feature Engineering | ✅ Advanced | Temporal features |
| Isolation Forest | ✅ Trained | Fast anomaly detection |
| LSTM Autoencoder | ✅ Trained | Deep learning approach |
| Evaluation | ✅ Detailed | Metrics + visualizations |
| Documentation | ✅ Thorough | README + notebook |
| Business Insights | ✅ Actionable | Real-world applications |

---

**🌟 This project showcases a complete machine learning pipeline from problem understanding to deployment-ready solution.**
