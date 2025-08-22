# PetroNus: Well Log Analysis Hub

**PetroNus** is a Streamlit-based UI for predicting payzones, porosity (phi), and water saturation (Sw) from well log CSV files. The UI allows easy visualization of predicted payzones and comparison of actual vs. predicted phi and Sw values.

## Features
- Upload multiple well CSV files for analysis.
- Predict payzones using a pre-trained model (`extra_trees_model.pkl`).
- Predict porosity (`phi`) and water saturation (`Sw`) using pre-trained models (`phi_model.pkl` and `sw_model.pkl`).
- Interactive visualizations using Plotly.
- Summary metrics like total payzones, payzone percentage, and total records.

## Installation

1. Clone this repository:
   ```bash
   git clone <your-repo-url>
   cd PetroNus
