# Ad Click-Through Rate (CTR) Prediction using Logistic Regression

## Project Overview
This project predicts whether an online user will click on an advertisement based on demographic and behavioral attributes such as age, area income, daily internet usage, and daily time spent on site.

Using **Logistic Regression**, the model classifies users into:
- `1` → User likely to click an ad  
- `0` → User not likely to click  

The project demonstrates practical application of machine learning for digital marketing optimization.

## Objectives
- Perform data analysis and preprocessing on ad interaction data.
- Train a **Logistic Regression** model for binary classification.
- Evaluate performance using **Accuracy, Precision, Recall, F1-score**, and **Confusion Matrix**.
- Identify key factors influencing ad clicks.

## Model Performance
| Metric | Score |
|--------|--------|
| Accuracy | 0.98 |
| Precision | 0.989 |
| Recall | 0.97 |
| F1-Score | 0.98 |

The confusion matrix visualization confirmed minimal misclassifications.

## Methodology
1. **Data Preprocessing**
   - Dropped irrelevant columns (`Ad Topic Line`, `City`, `Country`, `Timestamp`).
   - Standardized features using `StandardScaler`.
   - Split data into 80% training and 20% testing sets.

2. **Model Building**
   - Implemented **Logistic Regression** using `scikit-learn`.
   - Trained with balanced class weights.

3. **Evaluation**
   - Evaluated with Accuracy, Precision, Recall, F1-score.
   - Visualized with a Confusion Matrix.

## Tools and Libraries
| Library | Purpose |
|----------|----------|
| pandas, numpy | Data handling |
| matplotlib, seaborn | Visualization |
| scikit-learn | Model training & metrics |
| joblib | Model saving |
| jupyter | Notebook environment |

## Results
- Logistic Regression achieved 98% accuracy.
- Strong correlation observed between Age, Daily Internet Usage, and Click Behavior.
- The model can be used by advertisers to predict user engagement and optimize targeting.

## Author
Name: Thotakura Anudeepthi
Project: Artificial Intelligence Minor Project
Institution:  KL UNIVERSITY 
