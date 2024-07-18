# Heart Disease Prediction Model

## Overview
This project focuses on developing a predictive model for early detection of heart failure using machine learning techniques. By analyzing clinical and demographic data of patients, the goal is to identify patterns and risk factors associated with heart failure. The project includes exploratory data analysis (EDA) to better understand the dataset structure and characteristics, followed by training and evaluation of multiple machine learning models. The ultimate objective is to select the model that offers the highest accuracy and robustness for predicting heart failure, optimizing hyperparameters through Grid Search.

## Market Problem
Early detection of heart failure aims to reduce medical costs and improve quality of life. Early prevention allows for interventions that can enhance patient longevity and overall well-being.

## Dataset
The dataset used in this project contains medical records sourced from two Kaggle datasets:
- [`heart_disease_uci.csv`] (https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data/data)
- [`heart.csv`](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data)

These datasets provide attributes such as age, cholesterol levels, maximum heart rate, and other relevant factors that can help in predicting heart disease.

## Project Overview
This repository includes the following directories:

* [data_sample](https://github.com/BrendzRdgz/Proyecto_ML/tree/main/data_sample): Contains the datasets used and modified during the project.
* [img](https://github.com/BrendzRdgz/Proyecto_ML/tree/main/img): Includes all graphs generated during the EDA phase.
* [model](https://github.com/BrendzRdgz/Proyecto_ML/tree/main/model): Includes the saved model.
* [process](https://github.com/BrendzRdgz/Proyecto_ML/tree/main/process): Contains Jupyter notebooks for data cleaning (01_Cleaning.ipynb), exploratory data analysis (02_EDA.ipynb), dataset splitting and modeling * *(03_Split_Modeling.ipynb), and model testing (04_Models.ipynb).
* [results_notebook](https://github.com/BrendzRdgz/Proyecto_ML/tree/main/results_notebook): Includes the summary document detailing the project's process, conclusions, and final insights.
* [app](https://github.com/BrendzRdgz/Proyecto_ML/tree/main/app): Online predictor in ES using streamlit.

## Key Findings
- **EDA Insights:** Significant correlations between certain health indicators and heart disease risk.
- **Model Performance:** The Random Forest model achieved an accuracy of 91.6%, with further considerations on false negatives for potential improvement.

## Conclusion
This project aims not only to develop an effective predictive model but also to contribute to improving healthcare and patient quality of life through advanced data analysis and machine learning techniques. Implementing these models can have a significant impact on clinical practice and public health management.

## Future Considerations
- **Enhanced Features:** Incorporating additional health metrics or genetic data for more accurate predictions.
- **Real-time Application:** Integration into healthcare systems for real-time risk assessment and decision support.

## Author
Brenda Rodriguez

For more details, refer to the Jupyter notebooks (`*.ipynb`) and the saved model (`best_rf_model.pkl`) in this repository.