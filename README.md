# Heart Disease Prediction Model

## Introduction
Heart disease is a significant public health concern globally, contributing to a substantial number of deaths each year. Early detection and prevention are crucial for reducing mortality rates associated with cardiovascular diseases. This project focuses on developing a machine learning model to predict the likelihood of heart disease based on various health indicators.

## Dataset
The dataset used in this project contains medical records sourced from two Kaggle datasets:
- `heart_disease_uci.csv`
- `heart.csv`

These datasets provide attributes such as age, cholesterol levels, maximum heart rate, and other relevant factors that can help in predicting heart disease.

## Project Overview
This repository includes the following components:

- **Data Cleaning:** Initial processing of the datasets to handle missing values, standardize data formats, and prepare for analysis.
- **Exploratory Data Analysis (EDA):** Comprehensive analysis to understand the data distribution, relationships between variables, and insights into potential predictors of heart disease.
- **Feature Engineering:** Creation of new features and transformation of categorical variables using dummy encoding.
- **Modeling:** Training and evaluation of machine learning models including Logistic Regression, Decision Tree, Random Forest, XGBoost, LGBM, and CatBoost.
- **Model Evaluation:** Comparison of models using cross-validation techniques and optimization of hyperparameters to improve model performance.
- **Final Model Selection:** Identification and refinement of the best-performing model (Random Forest) based on evaluation metrics and test set performance.
- **Deployment:** Saving the finalized model (`best_rf_model.pkl`) for potential future use in predicting heart disease cases.

## Key Findings
- **EDA Insights:** Significant correlations between certain health indicators and heart disease risk.
- **Model Performance:** The Random Forest model achieved an accuracy of 91.6%, with further considerations on false negatives for potential improvement.

## Conclusion
This project underscores the importance of leveraging machine learning for early detection and prevention of heart disease. By utilizing predictive models, medical professionals can potentially intervene earlier, leading to better patient outcomes and reduced healthcare costs.

## Future Considerations
- **Enhanced Features:** Incorporating additional health metrics or genetic data for more accurate predictions.
- **Real-time Application:** Integration into healthcare systems for real-time risk assessment and decision support.

## Author
Brenda Rodriguez

For more details, refer to the Jupyter notebooks (`*.ipynb`) and the saved model (`best_rf_model.pkl`) in this repository.