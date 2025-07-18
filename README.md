# Big Mart Sales Prediction: An End-to-End Machine Learning Project

## 1. Project Overview

This project aims to predict the `Item_Outlet_Sales` for various products across different Big Mart outlets. The goal is to build a robust regression model that can accurately forecast sales, leveraging various features related to products and stores. This repository documents the step-by-step process, from data preprocessing and feature engineering to advanced ensemble modeling and hyperparameter tuning.

## 2. Dataset

The dataset used for this project contains sales data for 1559 products across 10 outlets in different cities. It includes product details (e.g., `Item_Identifier`, `Item_Weight`, `Item_Fat_Content`, `Item_Visibility`, `Item_Type`, `Item_MRP`) and outlet details (e.g., `Outlet_Identifier`, `Outlet_Establishment_Year`, `Outlet_Size`, `Outlet_Location_Type`, `Outlet_Type`). The target variable for prediction is `Item_Outlet_Sales`.

## 3. Methodology

The project followed a structured machine learning pipeline:

### 3.1 Data Preprocessing

* **Missing Value Imputation:**
    * `Item_Weight`: Imputed using the mean weight of all items.
    * `Outlet_Size`: Imputed based on the most frequent `Outlet_Size` for each `Outlet_Type`.
* **Feature Creation:**
    * `Outlet_Age`: Calculated from `Outlet_Establishment_Year` (e.g., `current_year - Outlet_Establishment_Year`).
    * `Item_MRP_Category`: Categorized `Item_MRP` into bins (Low, Mid, High, Very High) to capture price segments.
* **Data Standardization:**
    * `Item_Fat_Content`: Standardized inconsistent entries (e.g., 'low fat', 'LF' to 'Low Fat'; 'reg' to 'Regular').
* **Target and Feature Transformation:**
    * `Item_Visibility_log`: Applied `log1p` transformation to `Item_Visibility` to handle its skewed distribution.
    * `Item_Outlet_Sales_sqrt`: Applied square root transformation to the target variable `Item_Outlet_Sales` to normalize its distribution and improve model performance.
* **Categorical Encoding:**
    * All identified categorical features (including newly created ones like `Item_MRP_Category` and `Outlet_Establishment_Year` treated as categorical) were converted into numerical format using One-Hot Encoding (`pd.get_dummies` with `drop_first=True` to avoid multicollinearity).
* **Column Dropping:** Original identifier columns (`Item_Identifier`, `Outlet_Identifier`) and features replaced by transformations (`Item_Visibility`, `Item_Outlet_Sales`) were dropped.

### 3.2 Feature Engineering (Initial Attempts)

Initial attempts were made to create additional interaction and aggregated features (e.g., `Item_MRP_Category_Outlet_Type_Interaction`, `Outlet_Age_Outlet_Type_Interaction`, `Item_Visibility_MRP_Interaction`, `Outlet_Type_Mean_MRP`, `Item_Type_Mean_MRP`). However, these did not lead to performance improvements and were consequently excluded from the final best feature set to maintain model simplicity and performance.

### 3.3 Model Selection & Hyperparameter Tuning

Various regression models were considered, with a strong focus on tree-based ensembles due to their performance on tabular data.

* **XGBoost Regressor:**
    * Initial model showed strong performance.
    * Extensive hyperparameter tuning was performed using `RandomizedSearchCV` to optimize its parameters (`colsample_bytree`, `gamma`, `learning_rate`, `max_depth`, `n_estimators`, `reg_alpha`, `subsample`).
* **Random Forest Regressor:**
    * Utilized as a diverse tree-based ensemble.
    * Hyperparameter tuned using `RandomizedSearchCV` (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `bootstrap`).
* **LightGBM Regressor:**
    * Another high-performance gradient boosting library.
    * Hyperparameter tuned using `RandomizedSearchCV` (`n_estimators`, `learning_rate`, `num_leaves`, `max_depth`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`).

### 3.4 Ensemble Modeling (Stacking)

To further enhance predictive power and leverage the strengths of individual models, a `StackingRegressor` was implemented:

* **Base Models (Level-0):**
    * Hyper-tuned XGBoost Regressor
    * Hyper-tuned LightGBM Regressor
    * Hyper-tuned Random Forest Regressor
    * Ridge Regressor (a simpler linear model for diversity)
* **Meta-Model (Level-1):**
    * Ridge Regressor (chosen for its simplicity and effectiveness in combining predictions).
* **Training:** The stacking model was trained using 5-fold cross-validation (`cv=5`) to generate out-of-fold predictions for the meta-model, reducing overfitting.

## 4. Key Results

The model performance was evaluated using **R-squared**, **Mean Absolute Error (MAE)**, and **Root Mean Squared Error (RMSE)** on the test set.

Here's a summary of the progression to the best model:

| Metric            | Initial Tuned XGBoost (Test) | Stacked (Non-Tuned Bases) (Test) | **Stacked (TUNED Bases)** (Test) |
| :---------------- | :--------------------------- | :------------------------------- | :---------------------------------------------- |
| **R-squared** | 0.6880                       | 0.6899                           | **0.6911** |
| **MAE** | 7.7746                       | 7.7346                           | **7.7260** |
| **RMSE** | 10.1258                      | 10.0946                          | **10.0763** |

The final **Stacked Regressor Model with Tuned Base Models** achieved the highest R-squared and lowest MAE/RMSE, demonstrating consistent improvements throughout the optimization process. The model also maintained **excellent generalization**, with minimal difference between training and testing scores, indicating no overfitting.

## 5. Visualizations

Several plots were generated to analyze the best model's performance:

* **Actual vs. Predicted Plot:** Showed a strong linear relationship between actual and predicted sales, with points clustering closely around the perfect prediction line.
* **Residuals Plot:** Indicated a random scatter of residuals around zero, suggesting the model captures most of the underlying patterns and does not exhibit systematic errors.
* **Distribution of Residuals:** Showed a nearly normal distribution of errors centered around zero, which is ideal.
* **Feature Importance Plot (from LightGBM base model):** Provided insights into which features were most influential in driving sales predictions.


## 6. How to Run

To replicate the results or run the code:

**Prerequisites:**
   - Python 3.x (Ensure that you have python 3.10 version to avoid library or package compatability issue)
   - Pip (Python package installer) (When you create virtual environment or install python, by default it installs pip and setuptools packages)

1. **Create a virtual environment (recommended):**
     ```bash
     python -m venv venv
     # On Windows
     .\venv\Scripts\activate
     # On macOS/Linux
     source venv/bin/activate
     ```

3.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Sonjoy95/Big-Mart-Sales-Prediction-Project.git
    cd Big-Mart-Sales-Prediction-Project
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Place Data:** Ensure `train.csv` is in the project's root directory.

7.  **Execute Code:** Run the Python scripts sequentially as discussed (preprocessing, tuning individual models, then stacking). The Jupyter Notebook or Colab notebook containing the full code would be ideal.

## 7. Dependencies

* `pandas=2.2.3`
* `numpy=2.2.6`
* `scipy=1.15.3`
* `scikit-learn=1.6.1`
* `xgboost=3.0.2`
* `lightgbm=4.6.0`
* `matplotlib=3.10.3`
* `seaborn=0.13.2`


## 8. Conclusion

This project successfully developed a highly accurate and robust model for Big Mart Sales prediction. Through meticulous data preprocessing, strategic feature engineering, thorough hyperparameter tuning, and advanced ensemble techniques like stacking, the model achieved state-of-the-art performance. This systematic approach can be applied to similar regression problems in other domains.
