# AI Job Market Salary Prediction

## Overview
This R project is focused on predicting AI-related job salaries using various machine learning models. The project leverages the `ai_job_market_insights.csv` dataset, which includes features like job title, industry, experience level, and location to predict salaries. The goal is to experiment with multiple models, including linear and non-linear approaches, to identify the most accurate salary prediction model.

## Key Features
- **Data Preprocessing**: 
  - Missing values handling using `na.omit()`
  - Box-Cox transformation to normalize the salary distribution
  - Dummy variable creation for categorical variables
  - Near-zero variance feature removal
  - Centering and scaling of numeric features for model training

- **Modeling**:
  - **Linear Models**: Linear Regression, Ridge, Lasso, and Elastic Net.
  - **Non-linear Models**: Neural Networks, MARS (Multivariate Adaptive Regression Splines), Support Vector Machines (SVM), and K-Nearest Neighbors (KNN).
  
- **Evaluation**:
  - Model evaluation metrics: RMSE (Root Mean Squared Error), R-squared, MAE (Mean Absolute Error).
  - Feature importance visualizations using `varImp()`.

- **Visualization**:
  - Visualize the distribution of salary before and after Box-Cox transformation.
  - Compare model performance using various metrics.

## Dataset
**Categorical Variables:**
- Job Title: Titles associated with various AI roles (e.g., Data Scientist, AI Engineer).
- Industry: The sectors employing AI professionals (e.g., Tech, Healthcare).
- Experience Level: Categorization based on the required experience (e.g., Entry-Level, Mid-Level).
- Location: Geographic region of employment.
- Education: Minimum educational qualifications (e.g., Bachelor’s, Master’s).
- Company Size: Size of the employing company (e.g., Small, Medium, Large).
- Remote Work: Proportion of work performed remotely.
- Job Posting Date: Period when the job was posted.
- Hiring Status: Current hiring stage (e.g., Open, Closed).
  
**Target Variable:**
- Salary (USD): Annual compensation for the AI roles.

## Required Libraries
- `dplyr`
- `caret`
- `ggplot2`
- `glmnet`
- `earth`
- `kernlab`
- `nnet`
- `gridExtra`
- `e1071`

To install these libraries, run the following R command:
```R
install.packages(c('dplyr', 'caret', 'ggplot2', 'glmnet', 'earth', 'kernlab', 'nnet', 'gridExtra', 'e1071'))
```
## Results and Evaluation
- **Linear Models**:
  - Linear Regression, Ridge Regression, Lasso Regression, and Elastic Net are trained and evaluated.
  
- **Non-linear Models**:
  - Neural Networks, MARS, SVM, and KNN are also trained and evaluated.
  
- **Model Performance**:
  - The script outputs evaluation metrics like RMSE, R-squared, and MAE for each model.

- **Best Model**:
  - The model performance metrics will indicate which model performs best for salary prediction.

## Future Work
- Experiment with ensemble models such as Random Forest or XGBoost.
- Perform additional feature engineering to improve model accuracy.
- Try hyperparameter tuning to optimize model performance further.
