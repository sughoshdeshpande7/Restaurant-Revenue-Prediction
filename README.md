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

The dataset is from the [Restaurant Revenue Prediction](https://www.kaggle.com/c/restaurant-revenue-prediction).

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

## 🚧 Limitations

### Limited Feature Set:
- The dataset primarily includes operational and contextual data, which may not capture all factors influencing restaurant revenue (e.g., economic trends, competition, or customer preferences).

### Data Quality Issues:
- Potential data inconsistencies such as missing values, noisy categorical labels, and limited variability in features may impact the model's ability to generalize.

### Assumptions in Feature Engineering:
- Engineered features like `Days Since Opening` assume a static snapshot, ignoring potential seasonal effects or trends over time.

### Overfitting Risks:
- Some models, especially ensemble methods like XGBoost and LightGBM, are prone to overfitting when hyperparameters are not carefully tuned.

### Evaluation Metrics:
- The project primarily focuses on RMSE. Other business-relevant metrics (e.g., revenue percent error) may provide more actionable insights.

### Test Dataset Labels Not Available:
- Without actual revenue values for the test set, we rely solely on leaderboard scores, which limits comprehensive evaluation of model performance.

---

## 🚀 Future Work

### Incorporate External Data:
- Augment the dataset with external factors such as:
  - **Economic Indicators**: GDP, inflation, and disposable income levels.
  - **Geographic Data**: Restaurant proximity to competitors or population density.
  - **Weather Data**: Climate factors that could affect foot traffic.

### Time Series Modeling:
- Transition from static modeling to time series forecasting to capture seasonal patterns, growth trends, and temporal dependencies.

### Advanced Feature Engineering:
- Create interaction features or latent embeddings to capture complex relationships between variables.
- Use techniques like **Principal Component Analysis (PCA)** or **Autoencoders** for dimensionality reduction.

### Explore Other Evaluation Metrics:
- Introduce metrics like **Mean Absolute Percentage Error (MAPE)** or **Revenue Deviation** to better align evaluation with real-world business goals.

### Hyperparameter Optimization:
- Perform an extensive search (e.g., **Bayesian Optimization**) for optimal hyperparameters to further improve model performance.

### Model Interpretability:
- Use tools like **SHAP** or **LIME** to explain model predictions and identify key drivers of revenue.

### Deploy a Real-World Application:
- Build a web-based dashboard or API to predict restaurant revenues dynamically and visualize key insights.

### Ensemble with Deep Learning:
- Integrate neural networks or pre-trained embeddings (e.g., for location data) into the stacking model for better generalization.

