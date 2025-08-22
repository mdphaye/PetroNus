# pages/2_Phi_Sw_Predictor.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import plotly.graph_objects as go

st.set_page_config(page_title="Phi & Sw Predictor", layout="wide")

# Sidebar styling: light beige-brown background and darkest shade title color
st.markdown("""
<style>
    section[data-testid="stSidebar"] {
        background-color: #d0c7b6 !important;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] h5,
    section[data-testid="stSidebar"] h6 {
        color: #978f86 !important;
    }

    /* Existing minimal styling */
    .results-container {
        background: #e8f4f8;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #bee5eb;
        margin: 1rem 0;
    }
    .error-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    .stNumberInput > div > div > input::placeholder {
        color: #6c757d;
        opacity: 1;
    }
    /* Remove link icons from headings */
    .stMarkdown h1 .anchor-link,
    .stMarkdown h2 .anchor-link,
    .stMarkdown h3 .anchor-link,
    .stMarkdown h4 .anchor-link,
    .stMarkdown h5 .anchor-link,
    .stMarkdown h6 .anchor-link {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("Porosity(Phi) & Water Saturation(Sw) Prediction Tool")

# Load models
@st.cache_resource
def load_models_and_scaler():
    phi_model = joblib.load("phi_model.pkl")
    sw_model = joblib.load("sw_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return phi_model, sw_model, scaler

phi_model, sw_model, scaler = load_models_and_scaler()

# Input Form
st.markdown("### Input Well Log Parameters")

# File upload option
uploaded_file = st.file_uploader("Upload Well CSV File (Optional)", type="csv", help="Upload to auto-fill parameters from selected row")

well_data = None
selected_well_name = ""
if uploaded_file is not None:
    well_data = pd.read_csv(uploaded_file)
    selected_well_name = os.path.splitext(uploaded_file.name)[0]
    st.success(f"File uploaded successfully! {len(well_data)} rows available.")

with st.form("input_form"):
    well_id = st.text_input("Well Name / ID", 
                           value=selected_well_name if selected_well_name else "", 
                           placeholder="Enter well identifier")
    
    # Row selection if file is uploaded
    selected_row = None
    if well_data is not None:
        st.markdown("**Select Row from Uploaded Data**")
        row_index = st.selectbox("Choose row:", range(len(well_data)), format_func=lambda x: f"Row {x+1} - Depth: {well_data.iloc[x].get('DEPT', 'N/A')}")
        selected_row = well_data.iloc[row_index]
    
    st.markdown("**Log Parameters**")
    col1, col2 = st.columns(2)

    with col1:
        dept = st.number_input("DEPT (Depth)", 
                             value=float(selected_row.get('DEPT', 0)) if selected_row is not None else None, 
                             placeholder="866.5")
        rhoc = st.number_input("RHOC", 
                             value=float(selected_row.get('RHOC', 0)) if selected_row is not None else None, 
                             placeholder="2.45")
        gr = st.number_input("GR", 
                           value=float(selected_row.get('GR', 0)) if selected_row is not None else None, 
                           placeholder="75.3")
        rilm = st.number_input("RILM", 
                             value=float(selected_row.get('RILM', 0)) if selected_row is not None else None, 
                             placeholder="12.8")

    with col2:
        rll3 = st.number_input("RLL3", 
                             value=float(selected_row.get('RLL3', 0)) if selected_row is not None else None, 
                             placeholder="15.2")
        rild = st.number_input("RILD", 
                             value=float(selected_row.get('RILD', 0)) if selected_row is not None else None, 
                             placeholder="18.7")
        mn = st.number_input("MN", 
                           value=float(selected_row.get('MN', 0)) if selected_row is not None else None, 
                           placeholder="0.25")
        cnls = st.number_input("CNLS", 
                             value=float(selected_row.get('CNLS', 0)) if selected_row is not None else None, 
                             placeholder="0.18")

    st.markdown("**Actual Values (Optional - for error calculation)**")
    col3, col4 = st.columns(2)
    with col3:
        phi_actual = st.number_input("Phi (Actual)", 
                                   value=float(selected_row.get('phi', 0)) if selected_row is not None else None, 
                                   placeholder="0.15")
    with col4:
        sw_actual = st.number_input("Sw (Actual)", 
                                  value=float(selected_row.get('sw', 0)) if selected_row is not None else None, 
                                  placeholder="0.65")

    submitted = st.form_submit_button("Predict Phi & Sw")

# Results and Plots
if submitted:
    # Handle None values
    input_values = [dept or 0, rhoc or 0, gr or 0, rilm or 0, rll3 or 0, rild or 0, mn or 0, cnls or 0]
    
    input_data = pd.DataFrame([input_values], columns=['DEPT', 'RHOC', 'GR', 'RILM', 'RLL3', 'RILD', 'MN', 'CNLS'])

    # Scale and predict
    input_scaled = scaler.transform(input_data)
    phi_pred = phi_model.predict(input_scaled)[0]
    sw_pred = sw_model.predict(input_scaled)[0]

    st.success(f"Prediction completed for Well: {well_id}")

    # Results display
    st.markdown('<div class="results-container">', unsafe_allow_html=True)
    st.markdown("### Prediction Results")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Phi (Predicted)", f"{phi_pred:.4f}")
        if phi_actual and phi_actual > 0:
            st.metric("Phi (Actual)", f"{phi_actual:.4f}")
    with col2:
        st.metric("Sw (Predicted)", f"{sw_pred:.4f}")
        if sw_actual and sw_actual > 0:
            st.metric("Sw (Actual)", f"{sw_actual:.4f}")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Error metrics
    if (phi_actual and phi_actual > 0) or (sw_actual and sw_actual > 0):
        st.markdown('<div class="error-container">', unsafe_allow_html=True)
        st.markdown("### Accuracy Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if phi_actual and phi_actual > 0:
                phi_mae = mean_absolute_error([phi_actual], [phi_pred])
                phi_mse = mean_squared_error([phi_actual], [phi_pred])
                st.markdown("**Phi Errors**")
                st.write(f"MAE: {phi_mae:.4f}")
                st.write(f"MSE: {phi_mse:.4f}")
        
        with col2:
            if sw_actual and sw_actual > 0:
                sw_mae = mean_absolute_error([sw_actual], [sw_pred])
                sw_mse = mean_squared_error([sw_actual], [sw_pred])
                st.markdown("**Sw Errors**")
                st.write(f"MAE: {sw_mae:.4f}")
                st.write(f"MSE: {sw_mse:.4f}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Enter actual Phi and Sw values to view error metrics")

# Add Phi & Sw plots if full well data is available
if well_data is not None and len(well_data) > 1:
    required_cols = ['DEPT', 'RHOC', 'GR', 'RILM', 'RLL3', 'RILD', 'MN', 'CNLS', 'phi', 'sw']
    
    if all(col in well_data.columns for col in required_cols):
        st.markdown("---")
        st.markdown("### Well Analysis Plots")
        
        # Predict for entire well
        feature_cols = ['DEPT', 'RHOC', 'GR', 'RILM', 'RLL3', 'RILD', 'MN', 'CNLS']
        scaled_features = scaler.transform(well_data[feature_cols])
        well_data['phi_predicted'] = phi_model.predict(scaled_features)
        well_data['sw_predicted'] = sw_model.predict(scaled_features)
        
        # Sort and add buffer
        well_data_sorted = well_data.sort_values("DEPT")
        depth_min = well_data_sorted['DEPT'].min()
        depth_max = well_data_sorted['DEPT'].max()
        depth_step = well_data_sorted['DEPT'].diff().median() or 1

        buffer_above = pd.DataFrame()
        if depth_min > 0:
            buffer_above = pd.DataFrame({
                'DEPT': [depth_min - i * depth_step for i in range(2, 0, -1)],
                'phi': [None]*2, 'phi_predicted': [None]*2,
                'sw': [None]*2, 'sw_predicted': [None]*2
            })

        buffer_below = pd.DataFrame({
            'DEPT': [depth_max + i * depth_step for i in range(1, 3)],
            'phi': [None]*2, 'phi_predicted': [None]*2,
            'sw': [None]*2, 'sw_predicted': [None]*2
        })

        well_data_plot = pd.concat([buffer_above, well_data_sorted, buffer_below], ignore_index=True)
        well_data_plot = well_data_plot.sort_values("DEPT")
        depth_range = [well_data_plot['DEPT'].min(), well_data_plot['DEPT'].max()]

        # Plot ranges
        phi_min = min(well_data_plot['phi'].dropna().min(), well_data_plot['phi_predicted'].dropna().min())
        phi_max = max(well_data_plot['phi'].dropna().max(), well_data_plot['phi_predicted'].dropna().max())
        phi_range = [phi_min - 0.03, phi_max + 0.03]

        sw_min = min(well_data_plot['sw'].dropna().min(), well_data_plot['sw_predicted'].dropna().min())
        sw_max = max(well_data_plot['sw'].dropna().max(), well_data_plot['sw_predicted'].dropna().max())
        sw_range = [sw_min - 0.05, sw_max + 0.05]

        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        # Create combined subplot (1 row, 2 cols)
        fig = make_subplots(rows=1, cols=2)

        # Phi traces
        fig.add_trace(go.Scatter(
            x=well_data_plot['phi'], y=well_data_plot['DEPT'],
            mode='lines', name='Phi Actual', line=dict(color='blue', width=2)
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=well_data_plot['phi_predicted'], y=well_data_plot['DEPT'],
            mode='lines', name='Phi Predicted', line=dict(color='red', width=2, dash='dot')
        ), row=1, col=1)

        # Sw traces
        fig.add_trace(go.Scatter(
            x=well_data_plot['sw'], y=well_data_plot['DEPT'],
            mode='lines', name='Sw Actual', line=dict(color='green', width=2)
        ), row=1, col=2)

        fig.add_trace(go.Scatter(
            x=well_data_plot['sw_predicted'], y=well_data_plot['DEPT'],
            mode='lines', name='Sw Predicted', line=dict(color='orange', width=2, dash='dot')
        ), row=1, col=2)

        # Update x-axes like payzone page
        fig.update_xaxes(
            title_text="Porosity",
            title_standoff=20,
            range=phi_range,
            showgrid=True,
            gridcolor='lightgray',
            ticks="outside",
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror='ticks',
            tickfont=dict(size=12),
            row=1, col=1
        )

        fig.update_xaxes(
            title_text="Water Saturation",
            title_standoff=20,
            range=sw_range,
            showgrid=True,
            gridcolor='lightgray',
            ticks="outside",
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror='ticks',
            tickfont=dict(size=12),
            row=1, col=2
        )

        # Update y-axes like payzone page
        fig.update_yaxes(
            title_text="Depth (m)",
            autorange='reversed',
            showgrid=True,
            gridcolor='lightgray',
            ticks="outside",
            showline=True,
            linewidth=2,
            linecolor='black',
            tickfont=dict(size=12),
            row=1, col=1
        )
        fig.update_yaxes(
            autorange='reversed',
            showgrid=True,
            gridcolor='lightgray',
            ticks="outside",
            showline=True,
            linewidth=2,
            linecolor='black',
            tickfont=dict(size=12),
            showticklabels=True,
            row=1, col=2
        )

        # Layout and legend same as payzone page
        fig.update_layout(
            height=1000,
            margin=dict(l=50, r=50, t=80, b=80),
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(
                orientation="h",
                x=0.5,
                xanchor="center",
                y=1.08,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1
            ),
            dragmode=False
        )

        # Bottom static legend like payzone page
        fig.add_annotation(
            text=(
                "<span style='color:blue'><b>───</b></span> Phi Actual&nbsp;&nbsp;"
                "  "
                "<span style='color:red'><b>- - - - -</b></span> Phi Predicted&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
                "                                                                                                     "
                "<span style='color:green'><b>───</b></span> Sw Actual&nbsp;&nbsp;"
                "  "
                "<span style='color:orange'><b>- - - - -</b></span> Sw Predicted"
            ),
            xref="paper", yref="paper",
            x=0.5, y=-0.10,
            showarrow=False,
            font=dict(size=13, color="black"),
            align="center",
            bgcolor="rgba(255,255,255,0.8)",
            borderwidth=1
        )

        st.plotly_chart(fig, use_container_width=True, config={
            'displayModeBar': True,
            'modeBarButtonsToRemove': [
                'select2d', 'lasso2d', 'resetScale2d', 'autoScale2d'
            ],
            'displaylogo': False,
            'staticPlot': False,
            'responsive': True
        })
