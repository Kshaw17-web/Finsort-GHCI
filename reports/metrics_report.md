# FinSort Model Evaluation Report

## Dataset Summary

- **Training samples**: 2,000
- **Test samples**: 500
- **Number of classes**: 10

## Overall Performance

- **Macro F1 Score**: 1.0000

## Per-Class Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| airline | 1.0000 | 1.0000 | 1.0000 | 50 |
| coffee_shop | 1.0000 | 1.0000 | 1.0000 | 50 |
| ecommerce | 1.0000 | 1.0000 | 1.0000 | 50 |
| electronics | 1.0000 | 1.0000 | 1.0000 | 50 |
| entertainment | 1.0000 | 1.0000 | 1.0000 | 50 |
| fuel | 1.0000 | 1.0000 | 1.0000 | 50 |
| grocery | 1.0000 | 1.0000 | 1.0000 | 50 |
| pharmacy | 1.0000 | 1.0000 | 1.0000 | 50 |
| restaurant | 1.0000 | 1.0000 | 1.0000 | 50 |
| transport | 1.0000 | 1.0000 | 1.0000 | 50 |

### Macro Averages

- **Precision**: 1.0000
- **Recall**: 1.0000
- **F1-Score**: 1.0000

### Weighted Averages

- **Precision**: 1.0000
- **Recall**: 1.0000
- **F1-Score**: 1.0000

## Confusion Matrix

![Confusion Matrix](confusion_matrix.png)

## Summary

FinSort achieves a macro F1 score of **1.0000** on the test set of 500 samples, demonstrating strong performance across all 10 transaction categories. The model was trained on 2,000 samples using Logistic Regression with TF-IDF vectorization.