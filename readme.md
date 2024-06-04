# Heart Disease Prediction

This repository contains the code for predicting heart disease using a Logistic Regression model. The dataset used is the well-known heart disease dataset from the UCI Machine Learning Repository. The objective is to build a model that can accurately predict whether a patient has heart disease based on various medical attributes.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [How to Use](#how-to-use)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Heart disease is one of the leading causes of death worldwide. Early detection can save lives by enabling timely intervention. In this project, we use machine learning techniques to predict the presence of heart disease in patients using logistic regression.

## Dataset
The dataset used in this project is the heart disease dataset from the UCI Machine Learning Repository. It contains 303 instances and 14 attributes, including age, sex, chest pain type, resting blood pressure, cholesterol levels, and more.

## Exploratory Data Analysis
We begin by examining the data to understand its structure and the relationships between variables.

### Histograms of Features
![Histograms of Heart Disease Data Features](images/histograms.png)
*Histograms show the distribution of each feature in the dataset.*

### Correlation Matrix
![Correlation Matrix of Heart Disease Data](images/correlation_matrix.png)
*The correlation matrix helps identify relationships between features.*

## Model Training and Evaluation
We split the data into training and test sets, scale the features, and train a logistic regression model. The performance of the model is evaluated using accuracy and the ROC-AUC score.

### Accuracy
- **Training Data Accuracy**: Achieved an accuracy of 86.98% on the training data.
- **Test Data Accuracy**: Achieved an accuracy of 85.25% on the test data.

### ROC Curve
![ROC Curve](images/roc_curve.png)
*The ROC curve and AUC score help evaluate the model's performance in distinguishing between classes.*

## Results
Our logistic regression model shows a good performance with an AUC score of 0.93, indicating a high ability to distinguish between patients with and without heart disease.

## How to Use
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/heart-disease-prediction.git
