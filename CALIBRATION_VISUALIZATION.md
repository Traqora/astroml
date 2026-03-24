# Calibration Curve Visualization for Fraud Scores

## Overview

This module provides comprehensive calibration analysis tools for fraud detection models in the AstroML framework. Calibration curves help assess whether predicted fraud probabilities accurately reflect the true likelihood of fraudulent behavior.

---

## 🎯 **Why Calibration Matters**

### **Business Impact**
- **Risk Assessment**: Accurate probabilities enable better risk-based decisions
- **Regulatory Compliance**: Many regulations require well-calibrated risk scores
- **Operational Efficiency**: Proper calibration reduces false positives/negatives
- **Model Trust**: Stakeholders need reliable probability estimates

### **Technical Benefits**
- **Threshold Optimization**: Well-calibrated scores enable optimal threshold selection
- **Ensemble Methods**: Calibration improves model combination strategies
- **Cost-Sensitive Learning**: Accurate probabilities are essential for cost-sensitive applications

---

## 📊 **Key Metrics**

### **Primary Calibration Metrics**

| Metric | Range | Interpretation | Good Target |
|---------|-------|----------------|-------------|
| **Brier Score** | 0-1 | Overall accuracy + calibration | < 0.25 |
| **Log Loss** | 0-∞ | Probabilistic accuracy | < 0.5 |
| **Expected Calibration Error (ECE)** | 0-1 | Average calibration error | < 0.05 |

### **Confidence Metrics**

| Metric | Range | Interpretation |
|---------|-------|----------------|
| **Overconfidence** | 0-1 | Model too certain (predictions too extreme) |
| **Underconfidence** | 0-1 | Model too uncertain (predictions too conservative) |
| **Sharpness** | 0-∞ | Prediction variance (higher = more decisive) |

---

## 🛠️ **Usage Examples**

### **Basic Calibration Analysis**

```python
from astroml.validation.calibration import CalibrationAnalyzer

# Initialize analyzer
analyzer = CalibrationAnalyzer(n_bins=10, strategy='uniform')

# Compute calibration curve
fraction_pos, mean_pred = analyzer.compute_calibration_curve(y_true, y_prob)

# Generate comprehensive visualization
fig = analyzer.plot_calibration_curve(y_true, y_prob, "Fraud Detection Model")
plt.show()

# Generate detailed report
report = analyzer.generate_calibration_report(y_true, y_prob, "Fraud Detection Model")
print(report)
```

### **Multi-Model Comparison**

```python
# Compare multiple fraud detection models
models_data = {
    'Baseline Model': (y_true1, y_prob1),
    'Advanced Model': (y_true2, y_prob2),
    'Ensemble Model': (y_true3, y_prob3)
}

fig = analyzer.plot_multiple_models(models_data)
plt.show()
```

### **Calibration Improvement**

```python
# Apply temperature scaling for calibration improvement
temperature = 1.5
y_prob_calibrated = 1 / (1 + np.exp((np.log(y_prob / (1 - y_prob)) / temperature)))

# Compare before and after
models_data = {
    'Before Calibration': (y_true, y_prob),
    'After Calibration': (y_true, y_prob_calibrated)
}

fig = analyzer.plot_multiple_models(models_data)
plt.show()
```

---

## 📈 **Visualization Components**

### **1. Main Calibration Curve**
- **X-axis**: Mean predicted probability per bin
- **Y-axis**: Actual fraud rate per bin
- **Perfect Calibration**: Diagonal line (y = x)
- **Model Performance**: Deviation from diagonal

### **2. Prediction Distribution**
- **Green histogram**: Legitimate transactions
- **Red histogram**: Fraudulent transactions
- **Overlap**: Model discrimination ability

### **3. Reliability Diagram**
- **Bars**: Calibration per bin
- **Color intensity**: Sample count per bin
- **Reference line**: Perfect calibration

### **4. Metrics Summary**
- Comprehensive calibration metrics
- Sample statistics
- Interpretation guidelines

---

## 🔍 **Interpretation Guide**

### **Calibration Curve Patterns**

| Pattern | Interpretation | Action |
|---------|----------------|--------|
| **Close to diagonal** | Well-calibrated | No action needed |
| **Above diagonal** | Underconfident | Apply temperature scaling |
| **Below diagonal** | Overconfident | Apply Platt scaling |
| **S-shaped curve** | Systematic bias | Consider isotonic regression |

### **Common Issues & Solutions**

#### **Overconfidence**
```python
# Symptoms: High overconfidence metric, curve below diagonal
# Solution: Temperature scaling with T > 1
temperature = 1.5
y_prob_calibrated = 1 / (1 + np.exp((np.log(y_prob / (1 - y_prob)) / temperature))
```

#### **Underconfidence**
```python
# Symptoms: High underconfidence metric, curve above diagonal  
# Solution: Temperature scaling with T < 1
temperature = 0.7
y_prob_calibrated = 1 / (1 + np.exp((np.log(y_prob / (1 - y_prob)) / temperature))
```

#### **Non-monotonic Calibration**
```python
# Symptoms: Complex calibration curve shape
# Solution: Isotonic regression
from sklearn.isotonic import IsotonicRegression
ir = IsotonicRegression(out_of_bounds='clip')
y_prob_calibrated = ir.fit_transform(y_prob, y_true)
```

---

## 📋 **Best Practices**

### **Data Requirements**
- **Minimum samples**: 1000+ per calibration bin
- **Fraud representation**: At least 50 fraud cases per bin
- **Time consistency**: Validate across different time periods

### **Model Development**
1. **Split data**: Train/validation/test with temporal split
2. **Calibrate on validation**: Apply calibration methods on validation set
3. **Test on holdout**: Evaluate calibration on unseen test data
4. **Monitor over time**: Track calibration drift in production

### **Production Monitoring**
```python
# Regular calibration checks
def monitor_calibration(model, new_data, threshold_ece=0.05):
    y_true, y_prob = model.predict_proba(new_data)
    ece = analyzer.compute_calibration_metrics(y_true, y_prob)['ece']
    
    if ece > threshold_ece:
        logger.warning(f"Calibration degradation detected: ECE = {ece:.3f}")
        return False
    return True
```

---

## 🧪 **Advanced Features**

### **Adaptive Binning**
```python
# Use quantile-based binning for imbalanced datasets
analyzer = CalibrationAnalyzer(n_bins=10, strategy='quantile')
```

### **Sample Weighting**
```python
# Account for different transaction values
sample_weights = transaction_amounts / np.mean(transaction_amounts)
fraction_pos, mean_pred = analyzer.compute_calibration_curve(
    y_true, y_prob, sample_weight=sample_weights
)
```

### **Confidence Intervals**
```python
# Calibration curves include statistical confidence intervals
fig = analyzer.plot_calibration_curve(y_true, y_prob, confidence_level=0.95)
```

---

## 📊 **Example Output**

### **Sample Calibration Report**

```
# Calibration Report for Advanced Fraud Model

## Summary Statistics
- Total Samples: 50,000
- Fraud Rate: 0.082 (8.2%)
- Mean Prediction: 0.095
- Prediction Range: [0.001, 0.998]

## Calibration Metrics

### Primary Metrics
- **Brier Score**: 0.0843 (Excellent)
- **Log Loss**: 0.2341 (Good)

### Calibration Error Metrics
- **Expected Calibration Error (ECE)**: 0.0234 (Good)
- **Maximum Calibration Error (MCE)**: 0.0891 (Fair)
- **Adaptive Calibration Error (ACE)**: 0.0312 (Good)

### Confidence Analysis
- **Overconfidence**: 0.0123 (Low)
- **Underconfidence**: 0.0000 (None)
- **Sharpness**: 0.0876 (Good)

## Recommendations
- Model calibration is good with minor overconfidence
- Consider slight temperature scaling for optimal performance
- Monitor ECE in production, retrain if > 0.05
```

---

## 🔧 **Integration with AstroML**

### **Model Training Pipeline**
```python
from astroml.validation.calibration import CalibrationAnalyzer

class FraudModelPipeline:
    def __init__(self):
        self.calibration_analyzer = CalibrationAnalyzer()
    
    def train_and_calibrate(self, X_train, y_train, X_val, y_val):
        # Train base model
        self.model = self.train_base_model(X_train, y_train)
        
        # Get validation predictions
        y_val_prob = self.model.predict_proba(X_val)[:, 1]
        
        # Check calibration
        ece = self.calibration_analyzer.compute_calibration_metrics(
            y_val, y_val_prob
        )['ece']
        
        # Apply calibration if needed
        if ece > 0.05:
            self.calibrator = self.fit_calibrator(y_val_prob, y_val)
        else:
            self.calibrator = None
    
    def predict_proba(self, X):
        y_prob = self.model.predict_proba(X)[:, 1]
        if self.calibrator:
            y_prob = self.calibrator.transform(y_prob)
        return y_prob
```

### **Production Monitoring**
```python
class ProductionMonitor:
    def __init__(self, model, calibration_threshold=0.05):
        self.model = model
        self.analyzer = CalibrationAnalyzer()
        self.threshold = calibration_threshold
    
    def check_model_health(self, recent_data):
        y_true, y_prob = self.model.predict_with_labels(recent_data)
        ece = self.analyzer.compute_calibration_metrics(y_true, y_prob)['ece']
        
        return {
            'calibration_healthy': ece < self.threshold,
            'current_ece': ece,
            'recommendation': 'Retrain' if ece >= self.threshold else 'Monitor'
        }
```

---

## 📚 **Technical References**

1. **Guo, C., et al.** "On Calibration of Modern Neural Networks"
2. **Naeini, M. P., et al.** "Obtaining Well Calibrated Probabilities Using Bayesian Binning"
3. **Kull, M., et al.** "Beyond Temperature Scaling: Obtaining Well-Calibrated Multi-Class Probabilities"

---

## 🚀 **Getting Started**

### **Installation**
```bash
# Calibration module is part of AstroML validation
from astroml.validation import calibration
```

### **Quick Start**
```python
# Run the complete example suite
python examples/calibration_example.py
```

### **Custom Analysis**
```python
# Create your own calibration analysis
from astroml.validation.calibration import CalibrationAnalyzer

analyzer = CalibrationAnalyzer()
fig = analyzer.plot_calibration_curve(y_true, y_prob, "My Model")
plt.show()
```

This calibration visualization system provides comprehensive tools for ensuring your fraud detection models produce reliable, trustworthy probability estimates that align with real-world outcomes.
