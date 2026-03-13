# Donor Retention Prediction using Machine Learning

## Project Overview

This project applies machine learning techniques to predict whether a donor will donate again in the following year.

The goal is to help organizations improve fundraising efficiency by targeting donors with the highest probability of responding instead of sending marketing campaigns to the entire donor base.

---

## Objective

Nonprofit organizations often spend significant resources on fundraising campaigns.
This project aims to develop a predictive model that identifies donors most likely to respond to future donation requests, allowing organizations to optimize their mailing strategies and reduce unnecessary costs.

---

## Dataset

The dataset contains historical donor information including donation frequency, recency, and monetary value.

Key attributes include:

* Recency of last donation
* Frequency of donations
* Monetary value of donations
* Historical donor activity

Feature engineering techniques were applied to create **RFM (Recency, Frequency, Monetary) variables** commonly used in marketing analytics.

---

## Tools & Technologies

* R
* tidyverse
* tidymodels
* ranger
* xgboost
* pROC
* R Markdown

---

## Methodology

1. Data Cleaning and Preparation
2. Feature Engineering using RFM variables
3. Train/Test Data Split
4. Model Training and Comparison
5. Model Evaluation using classification metrics
6. Marketing strategy simulation

Machine learning models evaluated:

* Logistic Regression
* Stepwise Logistic Regression
* Decision Tree
* Tuned Decision Tree
* Bagged Trees
* Random Forest
* Gradient Boosting (XGBoost)

---

## Results

Models were evaluated using the following metrics:

* AUC
* Sensitivity
* Specificity

The **Stepwise Logistic Regression** model achieved the best performance with an **AUC of 0.693**.

The results demonstrate how predictive modeling can improve fundraising campaign efficiency by identifying high-probability donors.

---

## Full Project Report

A detailed analysis including exploratory data analysis, modeling steps, and evaluation results can be found in the project report.

📄 **HTML Version**
report/donor_retention_report.html

📄 **PDF Version**
report/donor_retention_report.pdf

---

## Project Structure

```
donor-retention-prediction
│
├── data
├── code
├── report
├── README.md
```

---

## How to Run

1. Clone the repository

```
git clone https://github.com/goldteaa/donor-retention-prediction.git
```

2. Open the R Markdown file in RStudio

3. Run the analysis to reproduce the results.

---

## Future Improvements

* Hyperparameter tuning for tree-based models
* Cross-validation for more robust evaluation
* Model interpretability using SHAP values
* Deployment as a donor targeting decision tool
