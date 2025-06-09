# Projet_Final_MLSUP

Welcome to the GitHub repository for my final Machine Learning Project Work. In this project, I explore a variety of classification techniques to predict next-day rainfall by analysing historical weather data from Australia's Bureau of Meteorology.


## Table of Contents
- [Instructions](#instructions)
- [About the Data](#about-the-data)
- [Getting Started](#getting-started)
- [Data Preparation](#data-preparation)
- [Model Development](#model-development)
- [Submission Guidelines](#submission-guidelines)
- [Credits](#credits)

## Instructions
Throughout this project, I'll guide you through the process of applying the classification algorithms we've learnt about. We will build models using training data and then evaluate their performance on test data with metrics like accuracy, Jaccard index, F1-Score, and more.

## About the Data
The data we're using has been sourced from the Australian Government's Bureau of Meteorology, enriched with additional predictive features crucial for our analyses. You can find more detailed information about the data [here](http://www.bom.gov.au/climate/dwo/).

### Dataset Features
- **Period**: 2008 - 2017
- **Variables Include**: Temperature ranges, humidity levels, atmospheric pressure, rainfall, and more.

## Getting Started
First, make sure you have all the required libraries:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import jaccard_score, f1_score, log_loss, accuracy_score

# Load the dataset
df = pd.read_csv('Weather_Data.csv')
```

## Data Preparation
We'll prepare our dataset for the models by handling categorical variables and normalizing the data.

### One Hot Encoding
This process converts categorical variables such as 'WindGustDir' and 'RainToday' into a binary format which our models can better understand.

## Model Development
I'll take you through training various models and explain how to interpret each model's evaluation metrics.

- **Accuracy Score**
- **Jaccard Index**
- **F1-Score**
- **LogLoss** (specific to logistic regression)
- **Mean Absolute Error**
- **Mean Squared Error**
- **R2-Score**

We'll explore the following models:

1. **Linear Regression**
2. **KNN**
3. **Decision Trees**
4. **Logistic Regression**
5. **SVM**
