# Donor Retention Prediction

Author: Artea Myftiu  
Course: Predictive Analytics & Big Data

## Project Overview

This project uses machine learning in R to predict whether a donor will donate again in 2018.

The goal is to help organizations improve mailing strategy by targeting donors with the highest probability of responding, instead of mailing everyone.

## Models Used

- Logistic Regression
- Stepwise Logistic Regression
- Decision Tree
- Tuned Decision Tree
- Bagged Trees
- Random Forest
- Gradient Boosting (XGBoost)

## Evaluation Metrics

- AUC
- Sensitivity
- Specificity

## Key Result

The best-performing model was the stepwise logistic regression model, with an AUC of 0.693.

## Project Structure

- `data/` → raw and engineered datasets
- `code/` → R Markdown and R code
- `report/` → final HTML and PDF reports

## Tools Used

- R
- tidyverse
- tidymodels
- ranger
- xgboost
- pROC