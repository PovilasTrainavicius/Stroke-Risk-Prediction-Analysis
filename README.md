# Stroke Risk Prediction Analysis

## Table of Contents
1. **Data cleaning and preprocessing**
2. **EDA: analyzing features**
3. **Statistical inference**
4. **Feature Engineering**
5. **Machine learning models**
6. **Stroke risk prediction analysis results**

## Overview
According to the World Health Organization (WHO) stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths. This dataset is used to predict whether a patient is likely to get stroke based on the input parameters like gender, age, various diseases, and smoking status. 

## Goal of the Analysis
To reduce the incidence of stroke among patients at The Johns Hopkins Hospital by enabling proactive, data-driven preventative care.


## Objective
To develop a predictive model for The Johns Hopkins Hospital that allows for the early identification and stratification of high-risk individuals. This model will serve as a clinical support tool, allowing physicians to implement targeted interventions and provide personalized counseling to patients and their families.


## Context of the Data
The dataset includes the following features:
- **id**: A unique identifier for each entry.
- **gender**: The gender of the patient (Male, Female, or Other).
- **age**: The age of the patient.
- **hypertension**: Whether the patient has hypertension (0 for no, 1 for yes).
- **heart_disease**: Whether the patient has a heart disease (0 for no, 1 for yes).
- **ever_married**: Whether the patient has ever been married (No or Yes).
- **work_type**: The patient's employment status (children, Govt_job, Never_worked, Private, or Self-employed).
- **residence_type**: The type of area the patient resides in (Rural or Urban).
- **avg_glucose_level**: The average glucose level in the patient's blood.
- **bmi**: The patient's body mass index.
- **smoking_status**: The patient's smoking history (formerly smoked, never smoked, smokes, or Unknown*). 
- **stroke**: Whether the patient had a stroke (1 for yes, 0 for no).

`Note: Unknown means the information is unavailable for this patient.`

## Analysis Overview
The project includes the following steps:

- **Data cleaning and preprocessing**
- **EDA: analyzing features**
- **Feature Engineering**
- **Machine learning models**
- **Model Selection**
- **Stroke risk prediction analysis results**

## Results

The LGBMClassifier model showed the best overall performance with:
- **Precision**: 0.123
- **Recall**: 0.799
- **F1-Score**: 0.213
- **PR-AUC**: 0.226

After adjusting treshold (`8%`):
- **Precision**: 0.081
- **Recall**: 0.973
- **F1-Score**: 0.149
- **PR-AUC**: 0.21

After ensemble models:
- **Precision**: 0.081
- **Recall**: 0.973
- **F1-Score**: 0.149
- **PR-AUC**: 0.227


## Technologies Used
- **Python**
  - `Pandas`, `NumPy` — data manipulation and numerical operations
  - `Matplotlib`, `Seaborn` — data visualization
  - `Scikit-learn` — preprocessing, pipelines, model building, and evaluation
  - `Imbalanced-learn (imblearn)` — handling imbalanced datasets with pipelines and resampling
  - `XGBoost`, `LightGBM`, `CatBoost` — gradient boosting classifiers for model training
  - `SciPy` — statistical tests and probability distributions
- **Jupyter Notebook** — interactive development and analysis environment
- **Git & GitHub** — version control and collaboration


## How to Run

To run this project locally:

1. **Clone the repository**  
   ```bash
   git clone https://github.com/TuringCollegeSubmissions/ptrain-DS.v2.5.3.2.5
   
2. **Navigate to the project directory**
    ```bash
    cd ptrain-DS.v2.5.3.2.5

3. **Install the required packages**  
    ```bash
    pip install -r requirements.txt` 

4. **Open and run `stroke_risk_prediction.ipynb`** to see the full analysis.


## 🔗 Links
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/povilas-trainavičius-0163ab217/)