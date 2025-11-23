# FinSort Metrics Report

### 1. Overall Accuracy
**Accuracy: 0.96**

### 2. Macro F1 Score
**Macro F1: 0.96**

### 3. Per-Category Performance
Paste from evaluation.py:

- bills — 0.99  
- coffee_shop — 0.91  
- dining — 0.94  
- ecommerce — 0.93  
- electronics — 0.97  
- entertainment — 0.99  
- grocery — 1.00  
- pharmacy — 1.00  
- refund — 1.00  
- transport — 1.00  
- travel — 0.99  
- wallet — 0.96  

### 4. Confusion Matrix
Paste generated matrix (or attach screenshot).

### 5. Interpretation
- Excellent performance on ecommerce, bills, fuel, travel  
- Lower confidence on unseen merchants → mitigated using:
  - Rule-based merchant matching
  - Confidence thresholding
  - Feedback logging
