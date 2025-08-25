# Big Mart Sales Prediction: An End-to-End Machine Learning Project 📈

## 1. Project Overview

This project focuses on building a robust regression model to accurately predict `Item_Outlet_Sales` for various products at different Big Mart outlets. The goal is to create a comprehensive and reproducible solution that demonstrates an end-to-end machine learning pipeline, from data preprocessing and feature engineering to advanced ensemble modeling and hyperparameter tuning. The final solution is a highly accurate **Stacked Regressor Model**.

## 2. Dataset

This project utilizes the Big Mart Sales dataset, which contains sales data for 1559 products across 10 outlets. The dataset includes product-related features like `Item_Weight` and `Item_MRP`, as well as outlet-related features such as `Outlet_Size` and `Outlet_Location_Type`. The target variable is `Item_Outlet_Sales`.

***

## 3. Methodology

The project followed a systematic machine learning pipeline to ensure a robust and reliable model.

### 3.1 Data Preprocessing & Feature Engineering

* **Missing Value Imputation:** Missing values in `Item_Weight` were imputed with the mean weight of all items. `Outlet_Size` was imputed based on the mode (most frequent value) for each `Outlet_Type`.
* **Feature Creation:**
    * **`Outlet_Age`:** A new feature calculated from the `Outlet_Establishment_Year` to represent the age of each store.
    * **`Item_MRP_Category`:** The `Item_MRP` (price) was binned into four categories: Low, Mid, High, and Very High, to capture different price segments.
* **Data Standardization:** Inconsistent entries in `Item_Fat_Content` (e.g., 'low fat', 'LF', 'reg') were standardized to 'Low Fat' and 'Regular' for consistency.
* **Target and Feature Transformation:**
    * `Item_Visibility`: A **`log1p` transformation** was applied to handle the feature's high positive skewness.
    * `Item_Outlet_Sales`: The target variable was **square-root transformed** to normalize its distribution, which significantly improved model performance.
* **Pipeline Automation:** A `ColumnTransformer` was used to automate numerical scaling with `StandardScaler` and categorical encoding with `OneHotEncoder`, ensuring a clean and reproducible workflow.

***

### 3.2 Model Selection & Hyperparameter Tuning

Various tree-based regression models were selected for their proven performance on tabular data. Each model was rigorously tuned using **`RandomizedSearchCV`** to optimize its parameters and mitigate overfitting.

* **XGBoost Regressor:** An optimized version of the initial XGBoost model.
* **Random Forest Regressor:** A robust bagging ensemble model.
* **LightGBM Regressor:** A high-performance, gradient boosting framework.

### 3.3 Ensemble Modeling (Stacking)

To achieve the best possible performance, a **`StackingRegressor`** was implemented. This method combined the diverse predictions of multiple base models, with a meta-model learning how to best weigh their outputs.

* **Base Models:** Tuned versions of **XGBoost, LightGBM, Random Forest**, and a simple **Ridge Regressor** were used to provide diverse predictions.
* **Meta-Model:** A **Ridge Regressor** was used as the final estimator to combine the base models' predictions effectively.

***

## 4. Key Results

The models were evaluated using **R-squared**, **Mean Absolute Error (MAE)**, and **Root Mean Squared Error (RMSE)** on the test set.

| Metric | Initial Tuned XGBoost (Test) | Stacked (Non-Tuned Bases) (Test) | **Stacked (TUNED Bases)** (Test) |
| :--- | :--- | :--- | :--- |
| **R-squared** | 0.6132 | 0.6138 | **0.6158** |
| **MAE** | 708.35 | 709.32 | **708.18** |
| **RMSE** | 1025.27 | 1024.51 | **1021.85** |

The **Stacked Regressor Model with Tuned Base Models** delivered the highest R-squared and lowest error metrics, demonstrating its superior predictive power and excellent generalization to new data.

***

## 5. How to Run

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Sonjoy95/Big-Mart-Sales-Prediction-Project.git](https://github.com/Sonjoy95/Big-Mart-Sales-Prediction-Project.git)
    cd Big-Mart-Sales-Prediction-Project
    ```
2.  **Create and Activate a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Place Data:** Ensure the `train.csv` file is placed in the project's root directory.
5.  **Execute Code:** Run the Jupyter Notebook `BigMart_Sales_Prediction.ipynb` to see the full analysis and model building process.

***

## 6. Dependencies

* `pandas`
* `numpy`
* `scikit-learn`
* `xgboost`
* `lightgbm`
* `matplotlib`
* `seaborn`
* `joblib`

You can find the exact versions in `requirements.txt` to ensure full reproducibility.

***

## 7. Conclusion & Next Steps

This project successfully developed a highly accurate and robust model for Big Mart sales prediction. Through meticulous data preprocessing, strategic feature engineering, thorough hyperparameter tuning, and advanced ensemble techniques like stacking, the model achieved a high R-squared score and excellent generalization.

**Next Steps for Future Work:**
* **Feature Engineering:** Explore more complex feature interactions (e.g., using polynomial features) or time-series features.
* **Advanced Models:** Experiment with deep learning models, such as neural networks, to see if they can capture more complex patterns.
* **Deployment:** Containerize the solution using Docker and deploy the model as a web service (e.g., using Flask or FastAPI) to make predictions accessible via an API.
