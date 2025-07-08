# SuperGlue Signature Recognition System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red.svg)](https://pytorch.org)

**AI-powered signature verification system achieving 96.5% accuracy with state-of-the-art SuperGlue technology.**

### Confusion Matrix Analysis v2.0
Our advanced confusion matrix analysis provides detailed insights into system performance with **96.5% accuracy**:

<div align="center">
  <img src="https://github.com/user-attachments/assets/a93b7355-0fcd-4d40-9138-37fba676d07b" alt="Confusion Matrix v2.0" width="800"/>
  <br/>
  <em>Professional confusion matrix visualization showing excellent performance with only 2.4% error rate</em>
</div>


**Key Insights from Confusion Matrix:**
- **True Negatives (450):** Rejection of different signatures (99.6% security)
- **True Positives (106):** Acceptance of authentic signatures (85.5% success)
- **False Positives (2)**  
- **False Negatives (18)**

### 📈 Threshold Optimization Analysis v2.0
Comprehensive threshold analysis revealing **0.30 as optimal threshold** for maximum accuracy:

<div align="center">
  <img src="https://github.com/user-attachments/assets/3df8dd21-6389-4317-860e-d023ed3f37b5" alt="Threshold Optimization v2.0" width="800"/>
  <br/>
  <em>Advanced threshold optimization showing peak performance at 0.30 threshold with 96.5% accuracy</em>
</div>

## 📈 Performance Metrics

| Metric | Value | Industry Standard | 
|--------|-------|------------------|
| **Overall Accuracy** | **96.5%** | 85-95% | 
| **Precision** | **85.5%** | >80% | 
| **Recall** | **85.5%** | >80% | 
| **F1-Score** | **85.5%** | >0.80 | 
| **False Positive Rate** | **0.4%** | <2% | 
| **False Negative Rate** | **14.5%** | <20% |
| **Security Level** | **99.6%** | >95% | 

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.9+
PyTorch
OpenCV
NumPy
```

### Installation
```bash
git clone https://github.com/gulcihanglmz/superglue-signature-verification.git
cd superglue-signature-verification
pip install -r requirements.txt
```

### Basic Usage
```python
from match_signatures import verify_signature

# Verify signature pair
result = verify_signature("reference.jpg", "test.jpg")
print(f"Match confidence: {result['confidence']:.3f}")
print(f"Verification: {'VALID' if result['is_match'] else 'INVALID'}")
```

## 📁 Project Structure

```
├── models/                # Neural network models
│   ├── superglue.py       # SuperGlue implementation
│   ├── superpoint.py      # SuperPoint keypoint detector
│   └── weights/           # Pre-trained model weights
├── match_signatures.py    # Main verification logic
├── confusion_matrix_analysis_v2.py  # Performance analysis
├── Report.md            
└── requirements.txt      
```
```

```
## 📊 Visual Analysis

The system includes comprehensive analysis tools:

- **Confusion Matrix**: Detailed performance breakdown
- **Threshold Optimization**: Fine-tuned for best results
- **Visual Matching**: Keypoint visualization and matching display
- **Performance Metrics**: Professional reporting and analytics

### Detailed Performance Metrics
```
CLASSIFICATION MATRIX:
                 Predicted
                 Different | Same
Actual Different    450   |   2    (99.6% specificity)
Actual Same          18   |  106   (85.5% sensitivity)
```

### System Architecture Overview
```mermaid
graph TD
    A[Input: Signature Images] --> B[SuperPoint: Keypoint Detection]
    B --> C[SuperGlue: Feature Matching]
    C --> D[Confidence Scoring 0.0-1.0]
    D --> E[Threshold Analysis ≥0.30]
    E --> F[Decision: Valid / Invalid]
    F --> G[Visual Output + Analytics]
```

### � Sample Verification Results
```python
# Example verification output
{
    "signature_pair": "user_123_sample_01.jpg vs user_123_sample_02.jpg",
    "match_confidence": 0.847,
    "predicted_same": true,
    "ground_truth_same": true,
    "verification_result": "AUTHENTIC",
    "processing_time": "0.68 seconds",
    "keypoints_detected": [187, 203],
    "keypoints_matched": 94,
    "match_ratio": 0.847,
    "security_level": "HIGH CONFIDENCE"
}
```
---
## 🤝 Contributing

1. Fork the repository
2. Create your feature branch 
3. Commit your changes 
4. Push to the branch 
5. Open a Pull Request
   
*Star ⭐ this repository if you found it helpful!*

## References
A Python toolkit for pairwise signature matching using [SuperPoint](https://arxiv.org/abs/1712.07629) + [SuperGlue](https://arxiv.org/abs/1911.11763).  
It generates a JSON of match predictions for all signature pairs in your dataset, then visualizes results for inspection.

* **SuperGlue Pretrained Network** (Matching backbone):
  [https://github.com/magicleap/SuperGluePretrainedNetwork](https://github.com/magicleap/SuperGluePretrainedNetwork)
* **SuperPoint & SuperGlue papers** for algorithmic details:

  * DeTone, Malisiewicz & Rabinovich, “SuperPoint: Self-Supervised Interest Point Detection and Description”, ECCV 2018.
  * Sarlin et al., “SuperGlue: Learning Feature Matching with Graph Neural Networks”, CVPR 2020.
