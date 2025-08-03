# SuperGlue Signature Recognition System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red.svg)](https://pytorch.org)

**AI-powered signature verification system achieving 98.1% accuracy with state-of-the-art SuperGlue technology.**

### Confusion Matrix Analysis - Signature Recognition System
Our advanced confusion matrix analysis provides detailed insights into system performance with **98.1% accuracy**:

<div align="center">
  <img src="https://github.com/user-attachments/assets/cddf785e-0b67-4ac7-b842-9ef9d0ab8ab6" alt="Signature Recognition System Confusion Matrix" width="800"/>
  <br/>
  <em>Signature Recognition System confusion matrix with 1.9% error rate</em>
</div>

**Key Insights from Confusion Matrix:**
- **True Negatives (450):** Rejection of different signatures 
- **True Positives (115):** Acceptance of authentic signatures 
- **False Positives (2):** Minimal false acceptances 
- **False Negatives (9):** Low false rejections 

### 📈 Threshold Optimization Analysis v2.0
Comprehensive threshold analysis revealing **0.30 as optimal threshold** for maximum accuracy:

<div align="center">
  <img src="https://github.com/user-attachments/assets/3df8dd21-6389-4317-860e-d023ed3f37b5" alt="Threshold Optimization v2.0" width="800"/>
  <br/>
  <em>Advanced threshold optimization showing peak performance at 0.30 threshold with 98.1% accuracy</em>
</div>

### 📈 Results Examples
<div align="center">
  <img width="1542" height="563" alt="Screenshot 2025-07-31 141354" src="https://github.com/user-attachments/assets/a3c93c8e-a982-45b1-b6b1-dc509cd56d08" />
  <img width="1491" height="439" alt="Screenshot 2025-07-31 141849" src="https://github.com/user-attachments/assets/0b53d8e5-27e6-409f-9c20-78ba8c0801a4" />
  <img width="1565" height="569" alt="Screenshot 2025-07-31 142033" src="https://github.com/user-attachments/assets/d9b13982-6103-493c-89b6-b92f25fd34aa" />
</div>


## 📈 Performance Metrics

| Metric | Value | 
|--------|-------|
| **Overall Accuracy** | **98.1%** |
| **Precision** | **98.3%** |
| **Recall** | **92.7%** | 
| **F1-Score** | **95.4%** |
| **False Positive Rate** | 
| **False Negative Rate** | 

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
Actual Same           9   |  115   (92.7% sensitivity)
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
