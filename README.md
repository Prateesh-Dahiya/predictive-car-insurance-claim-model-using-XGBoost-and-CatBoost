# Car Insurance Claim Prediction Using Machine Learning

## 1. Project Overview

This project focuses on predicting car insurance claims using machine learning techniques. It addresses two predictive modelling problems:

- **Claim Classification:** Predict whether a policyholder is likely to make an insurance claim.
- **Claim Amount Prediction:** Estimate the monetary value of a claim, given that a claim has occurred.

The objective is to use customer demographics, vehicle characteristics, insurance policy details and historical claim information to build predictive models that can support insurance risk assessment and decision-making.

## 2. Dataset

The dataset is sourced from Kaggle:

[Car Insurance Claim Dataset](https://www.kaggle.com/datasets/xiaomengsun/car-insurance-claim-data)

**Dataset Overview:**

| Description | Details |
|---|---|
| Total Records | 10,302 |
| Total Features | 27 |
| Duplicate Records | 1 |
| Missing Values | 3,004 |
| Classification Target | `CLAIM_FLAG` |
| Regression Target | `CLM_AMT` |

The dataset contains information about:

- Customer demographics (age, income, gender, education and occupation)
- Vehicle characteristics (vehicle age, type and value)
- Insurance policy details
- Previous claim history
- Driving and licence-related information
- Claim occurrence and claim amount

## 3. Project Objectives

### Objective 1: Insurance Claim Prediction (Classification)

Predict whether a policyholder will make an insurance claim using the `CLAIM_FLAG` target variable.

### Objective 2: Insurance Claim Amount Prediction (Regression)

Predict the monetary value of an insurance claim using the `CLM_AMT` target variable for records where a claim has occurred.

## 4. Project Workflow

### Step 1: Data Cleaning and Preparation

- Inspected the dataset structure, data types and missing values.
- Identified and removed duplicate records.
- Converted currency-related columns into numerical format.
- Cleaned categorical values and renamed columns for better readability.
- Removed unnecessary features such as unique identifiers and redundant date information.

### Step 2: Exploratory Data Analysis (EDA)

- Analysed numerical and categorical features.
- Examined distributions and relationships between variables.
- Performed correlation analysis.
- Investigated patterns related to customer characteristics, vehicle information and previous claims.

### Step 3: Data Preprocessing

- Handled missing numerical values using KNN Imputation.
- Handled missing categorical values using most-frequent imputation.
- Applied Binary, Ordinal and One-Hot Encoding.
- Examined multicollinearity using Variance Inflation Factor (VIF).
- Used Scikit-learn preprocessing pipelines for data transformation.

### Step 4: Classification Model Development

Compared multiple classification algorithms:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Linear Support Vector Machine (SVM)
- Gradient Boosting
- AdaBoost
- Bagging
- XGBoost
- CatBoost

XGBoost was selected for further hyperparameter optimisation.

**Hyperparameter Optimisation:**
- RandomizedSearchCV
- GridSearchCV
- 5-fold cross-validation

**Evaluation Metric:**
- Weighted F1-score

### Step 5: Regression Model Development

Developed regression models to estimate claim amounts for policyholders with a claim.

The notebook compares multiple regression algorithms and uses the SGD Regressor for the final regression model.

**Optimisation Techniques:**
- RandomizedSearchCV
- GridSearchCV
- Cross-validation

**Evaluation Metrics:**
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

## 5. Model Performance

The following results are recorded in the notebook.

### Classification – XGBoost

| Evaluation Metric | Score |
|---|---:|
| Default Cross-Validation Weighted F1 | 0.7788 |
| Tuned Cross-Validation Weighted F1 | 0.7859 |
| Test Weighted F1-score | 0.7695 |

### Regression – SGD Regressor

| Evaluation Metric | Score |
|---|---:|
| Test RMSE | 8378.85 |
| Test MAE | 3556.26 |

*Note: These are the results recorded in the original notebook. They should be treated as preliminary until the complete preprocessing and evaluation workflow is revalidated.*

## 6. Technologies and Libraries

**Programming Language:** Python

**Libraries:**
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- CatBoost
- Statsmodels
- SciPy

**Development Environment:** Jupyter Notebook

## 7. Repository Structure

```text
Car-Insurance-Claim-Prediction/
│
├── main.ipynb
├── car_insurance_claim.csv
└── README.md
```

## 8. Installation and Execution

### Step 1: Clone the repository

```bash
git clone <your-github-repository-url>
```

### Step 2: Navigate to the project directory

```bash
cd Car-Insurance-Claim-Prediction
```

### Step 3: Install the required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost catboost statsmodels scipy jupyter
```

### Step 4: Open Jupyter Notebook

```bash
jupyter notebook
```

### Step 5: Run the notebook

Open `main.ipynb` and execute the cells sequentially.

Ensure that the dataset is available in the project directory and update the dataset path in the notebook if required.

## 9. Key Learnings

Through this project, I worked on:

- Data cleaning and exploratory data analysis.
- Handling missing values and categorical variables.
- Feature preprocessing and transformation pipelines.
- Classification and regression modelling.
- Comparing multiple machine learning algorithms.
- Hyperparameter optimisation using cross-validation.
- Evaluating predictive models using appropriate performance metrics.

## 10. Conclusion

This project demonstrates an end-to-end machine learning workflow for car insurance claim prediction and claim amount estimation. It combines exploratory data analysis, data preprocessing, classification, regression and hyperparameter optimisation to explore how historical policyholder information can be used for insurance claim analysis.

**Disclaimer:** This project is intended for educational and analytical purposes. Model performance on this dataset does not guarantee similar results on real-world insurance portfolios.
