# PetroNus 
## A Petrophysical Parameter Estimation and Payzone Detection Platform

## Overview
**PetroNus** is a Python-based Streamlit web application that predicts payzones, total porosity (phi), and water saturation (Sw) from well-log CSV files. It provides intuitive visualizations of predicted payzones and allows users to compare actual versus predicted phi and Sw values, enabling geoscientists and reservoir engineers to efficiently review and explore well log analyses.

<!-- ## Demo -->
<!-- ![PetroNus UI](UI/Petronus.gif) -->
# UI Demo
<img src="UI/Petronus.gif" alt="UI" width="900"/>


## Features
- Upload multiple well CSV files for analysis.
- Predict **payzones** using a pre-trained **Extra Trees Classifier** model.
- Predict **Phi** and **Sw** using pre-trained ensemble models with scaling.
- Interactive visualizations for:
  - Predicted payzones along depth
  - Actual vs predicted Phi (Porosity)
  - Actual vs predicted Sw (Water Saturation)
- Supports multi-well selection and analysis.

## Project Structure
```PetroNus/
├── main.py
├── pages/
│   ├── 1_Payzone_Pred.py
│   └── 2_Phi_Sw_Pred.py
├── extra_trees_model.pkl
├── phi_model.pkl
├── sw_model.pkl
├── scaler.pkl
├── README.md
└── requirements.txt
```

## Tools & Technologies Used
- **Programming Language:** Python 3.11  
- **Web App Framework:** Streamlit  
- **Data Handling & Analysis:** Pandas, NumPy  
- **Machine Learning & Modeling:** Scikit-learn, XGBoost, Optuna, TensorFlow, SciPy, Joblib  
- **Visualization:** Plotly, Matplotlib, Seaborn  
- **Version Control:** Git & GitHub  
- **Virtual Environment:** pyenv  
- **Platforms:** Jupyter Notebook, Streamlit, VS Code  
- **Skills Applied:** Data Cleaning, Data Preprocessing and Scaling, Feature Imputation, Feature Engineering, Regression Modeling, Outlier Removal, Reservoir Evaluation, Well Log Analysis, Payzone Visualization Plotting, UI Integration


## Project Workflow
- Data Upload: Upload one or more well CSV files via the sidebar.
- Data Parsing: The app reads and cleans the CSV files.
- Model Predictions:
   - Payzone prediction using Extra Trees Classifier.
   - Phi & Sw predictions using pre-trained ensemble models with scaling.

## Visualization
- Depth-aligned interactive plots for payzone, Phi, and Sw.
- Actual vs predicted comparisons.
- Summary Metrics: Total payzones, payzone percentage, and total records displayed.

## Notes & Recommendations
- The app is designed to work with pre-trained models only. You don’t need to retrain anything.
- Make sure all required columns exist in the uploaded CSV (DEPT, RHOC, GR, RILM, RLL3, RILD, MN, CNLS, phi, sw).
- Depth alignment and plot visualization are automatically adjusted per well.

## Contact
Mangalya D. Phaye – mphaye05@gmail.com – [LinkedIn](https://linkedin.com/in/mangalya-d-phaye-7883a4259) - [Github](https://github.com/mdphaye)