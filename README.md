# Restaurant Revenue Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Kaggle](https://img.shields.io/badge/Kaggle-Competition-brightgreen)

A machine learning solution for predicting restaurant revenues using advanced regression techniques, feature engineering, and model stacking. This project addresses challenges like feature selection, categorical encoding, and model evaluation with a focus on improving RMSE scores.

<p align="center">
  <img src="https://storage.googleapis.com/kaggle-media/competitions/kaggle/4272/media/TAB_banner2.png">
</p>

---

## 🌟 Overview

Predicting restaurant revenue is essential for optimizing business strategies and investments. This project aims to solve the [Restaurant Revenue Prediction](https://www.kaggle.com/c/restaurant-revenue-prediction) problem on Kaggle by leveraging a variety of machine learning models, including tree-based methods and ensemble techniques.

This project tackles challenges like noisy features, categorical data encoding, and feature importance ranking, while implementing advanced techniques like stacked model ensembling to improve predictive performance. All preprocessing, modeling, and evaluation were performed locally using Google Colab.

The final submission achieved a public leaderboard RMSE of 1.98M using CatBoost.

---

## 🔑 Key Features

### Preprocessing:
- Handling missing data, feature scaling, and one-hot encoding of categorical variables.
- Engineered new features, such as days since opening, year, and month, to improve predictive performance.

### Feature Selection:
- Used `SelectFromModel` (XGBoost), `SequentialFeatureSelector`, and `GeneticSelectionCV` to identify the most important features.

### Modeling:
- Built individual regression models, including XGBoost, LightGBM, Gradient Boosting, CatBoost, and more.
- Hyperparameter tuning for all models using Optuna.

### Stacking Ensemble:
- Combined predictions from the top-performing models using Ridge Regression as a meta-model to reduce RMSE further.

### Evaluation:
- Measured performance using Root Mean Squared Error (RMSE) and compared results across all models.
- Visualized feature importance, residuals, and prediction distributions for better insights.

---

## 📊 Dataset

The dataset contains historical revenue data for restaurants, along with features such as location, type, and operational details.

| **Column**      | **Description**                          |
|------------------|------------------------------------------|
| Open Date        | The date the restaurant opened          |
| City             | City where the restaurant is located    |
| City Group       | Type of city (Big City/Other)           |
| Type             | Type of restaurant (FC, IL, DT)         |
| Revenue          | Target variable: annual revenue         |

The dataset was split into training and test sets, with the goal of predicting **Revenue** for unseen test data.

---

## 🛠️ Project Workflow

### Exploratory Data Analysis (EDA):
- Analyzed revenue distributions, outliers, and relationships between features.
- Identified key features like Days Since Opening and categorical groupings.

### Data Preprocessing:
- Encoded categorical variables using `LabelEncoder` and One-Hot Encoding.
- Engineered new features (e.g., Year Opened, Days Since Open).
- Combined unseen labels in `City` to ensure consistency.

### Feature Selection:
- Identified important features using XGBoost and Genetic Algorithms.

### Model Training:
- Trained individual models, including:
  - **XGBoost**: Best individual model with RMSE of 0.6881.
  - LightGBM and Gradient Boosting: Complemented XGBoost.
  - CatBoost: Efficient handling of categorical data.
- Performed hyperparameter tuning using Optuna.

### Stacked Model:
- Combined predictions from XGBoost, LightGBM, and Gradient Boosting using Ridge Regression as a meta-model.
- Improved RMSE through ensemble learning.

### Submission:
- Generated final predictions for the test set.
- Submission CSVs for all models stored in `data/` for Kaggle evaluation.

---

## 📈 Results

| **Model**                 | **Private Score (RMSE)** | **Public Score (RMSE)** |
|----------------------------|--------------------------|--------------------------|
| Stacked Model (Selected Features) | 2.08M                   | 2.10M                   |
| XGBoost (Selected Features)       | 2.41M                   | 2.47M                   |
| Tuned Stacked Model               | 2.10M                   | 2.29M                   |
| Stacked Model                     | 2.38M                   | 2.68M                   |
| CatBoost                          | 2.10M                   | **1.98M**               |
| Decision Tree (DT)                | 2.94M                   | 3.17M                   |
| Elastic Net                       | 2.23M                   | 1.98M                   |
| Gradient Boosting (GB)            | 2.37M                   | 2.66M                   |
| LightGBM (LGB)                    | 2.29M                   | 2.16M                   |
| Multi-Layer Perceptron (MLP)      | 2.03M                   | 1.95M                   |
| Random Forest (RF)                | 2.11M                   | 2.09M                   |

**Leaderboard Score:** The best public score of **1.98M** was achieved with the **CatBoost model**, while the **Tuned Stacked Model** performed best on the private leaderboard.

---

## 📂 Files and Directories

| **File/Directory**         | **Description**                                      |
|-----------------------------|------------------------------------------------------|
| `data/train.csv`            | Training dataset with restaurant data and revenue.   |
| `data/test.csv`             | Test dataset for generating predictions.            |
| `data/xgboost_submission.csv` | Predictions from the XGBoost model.              |
| `notebooks/EDA.ipynb`       | Exploratory data analysis and initial insights.      |
| `src/preprocess.py`         | Preprocessing and feature engineering code.          |
| `src/train_models.py`       | Code to train and evaluate individual models.        |
| `src/stacking.py`           | Code to implement stacking for ensemble learning.    |

---

## 🧰 Dependencies

- Python 3.8+
- pandas, numpy
- xgboost, lightgbm, catboost
- scikit-learn
- matplotlib, seaborn
- mlxtend

---

## 📜 License

This project is licensed under the MIT License. See the LICENSE file for details.
