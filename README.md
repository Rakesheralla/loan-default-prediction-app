Home Loan Default Prediction System
Project Overview

This project is an end-to-end Machine Learning application designed to predict whether a customer is likely to default on a home loan.
It covers data analysis, model training, evaluation, and deployment using a simple web-based interface.

The trained model is deployed using Streamlit, allowing users to upload a test dataset and receive loan default predictions along with probability scores.

Problem Statement

Loan defaults pose a significant financial risk to banking institutions.
The objective of this project is to:

Analyze customer financial and credit data

Build a predictive model to identify high-risk loan applicants

Provide a batch prediction interface for real-world use

Dataset Description

The dataset is derived from the Home Credit loan default dataset and includes the following files:

application_train.csv – Main dataset containing the target variable

bureau.csv and bureau_balance.csv – Credit history from other institutions

previous_application.csv – Historical loan applications

POS_CASH_balance.csv – POS and cash loan history

credit_card_balance.csv – Credit card usage history

installments_payments.csv – Repayment behavior

Target Variable:

0 – Non-Defaulter

1 – Defaulter

Machine Learning Approach

Data cleaning and preprocessing

Exploratory Data Analysis (EDA)

Feature engineering

Handling class imbalance

Model training using Random Forest Classifier

Model evaluation

Probability-based decision making

A custom business threshold of 0.30 is applied instead of the default 0.50 to better handle the imbalanced nature of the dataset.

Application Features

Upload test dataset in CSV format

Automatic feature alignment

Loan default prediction

Default probability score for each record

Downloadable prediction results

Technology Stack

Python

Pandas, NumPy

Scikit-learn

Joblib

Streamlit

Jupyter Notebook