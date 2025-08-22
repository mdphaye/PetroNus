import os
import pandas as pd
import streamlit as st
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# === Page Setup ===
st.set_page_config(page_title="Payzone Predictor", layout="wide")

# Move Plotly toolbar below plots
st.markdown("""
<style>
    .modebar { top: auto !important; bottom: 0 !important; }
</style>
""", unsafe_allow_html=True)

# Sidebar styling
st.markdown("""
<style>
    section[data-testid="stSidebar"] {
        background-color: #d0c7b6 !important;
    }
    .metric-box {
        background: #f1f3f4;
        padding: 1rem;
        border-radius: 5px;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
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

st.sidebar.title("Payzone Predictor")

# === Load Models ===
@st.cache_resource
def load_payzone_model():
    return joblib.load("/Users/mangalyaphaye/Desktop/PetroNus/extra_trees_model.pkl")

@st.cache_resource
def load_phi_sw_models_and_scaler():
    phi_model = joblib.load("phi_model.pkl")
    sw_model = joblib.load("sw_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return phi_model, sw_model, scaler

et_model = load_payzone_model()
phi_model, sw_model, scaler = load_phi_sw_models_and_scaler()

# === Upload Files ===
st.sidebar.markdown("### Upload Files")
uploaded_files = st.sidebar.file_uploader(
    "Upload Well CSV Files",
    type="csv",
    accept_multiple_files=True
)

# === Parse CSV Files ===
well_dfs = {}
if uploaded_files:
    for file in uploaded_files:
        try:
            df = pd.read_csv(file)
            name = os.path.splitext(file.name)[0]
            well_dfs[name] = df
        except Exception as e:
            st.sidebar.error(f"Could not read {file.name}: {e}")

# === Sidebar Selection ===
selected_df = None
if well_dfs:
    st.sidebar.markdown("### Select Well")
    selected_well = st.sidebar.radio("Choose a well:", list(well_dfs.keys()))
    selected_df = well_dfs[selected_well]
    if selected_df is not None:
        st.sidebar.success(f"✓ {selected_well} loaded")

# === Main Area ===
st.title("Payzone Prediction Analysis")

if selected_df is not None:
    st.markdown(f"### Analyzing: **{selected_well}**")

    required_cols = ['DEPT', 'RHOC', 'GR', 'RILM', 'RLL3', 'RILD', 'MN', 'CNLS', 'phi', 'sw']

    if all(col in selected_df.columns for col in required_cols):
        # Predictions
        X_input = selected_df[required_cols]
        y_pred = et_model.predict(X_input)
        selected_df['PREDICTED_PAYZONE'] = y_pred

        feature_cols = ['DEPT', 'RHOC', 'GR', 'RILM', 'RLL3', 'RILD', 'MN', 'CNLS']
        scaled_features = scaler.transform(selected_df[feature_cols])
        selected_df['phi_predicted'] = phi_model.predict(scaled_features)
        selected_df['sw_predicted'] = sw_model.predict(scaled_features)

        # Summary metrics
        payzone_count = len(selected_df[selected_df['PREDICTED_PAYZONE'] == 1])
        total_count = len(selected_df)
        payzone_percentage = (payzone_count / total_count) * 100 if total_count else 0

        st.markdown("### Prediction Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Payzones", payzone_count)
        c2.metric("Payzone Percentage", f"{payzone_percentage:.1f}%")
        c3.metric("Total Records", total_count)

        # Depth alignment buffers
        selected_df = selected_df.sort_values("DEPT")
        depth_min, depth_max = selected_df['DEPT'].min(), selected_df['DEPT'].max()
        depth_range = [depth_min, depth_max]

        # Layout columns
        st.markdown("### Visualization")
        col1, col2 = st.columns([1.2, 2.5])

        # Payzone plot
        with col1:
            st.markdown("#### Predicted Payzones")
            st.markdown("""
                <style>
                    .your-plot-class {
                        margin-top: 40px;
                    }
                </style>
                """, unsafe_allow_html=True)

            fig_pay = go.Figure()
            payzones = selected_df[selected_df['PREDICTED_PAYZONE'] == 1]
        
            for _, row in payzones.iterrows():
                fig_pay.add_trace(go.Scatter(
                    x=[0, 1], y=[row['DEPT'], row['DEPT']],
                    mode="lines",
                    line=dict(color="green", width=2),
                    hovertemplate="<br>".join([
                        f"<b>DEPT:</b> {row['DEPT']}",
                        f"RHOC: {row.get('RHOC', '')}",
                        f"GR: {row.get('GR', '')}",
                        f"RILM: {row.get('RILM', '')}",
                        f"RLL3: {row.get('RLL3', '')}",
                        f"RILD: {row.get('RILD', '')}",
                        f"MN: {row.get('MN', '')}",
                        f"CNLS: {row.get('CNLS', '')}",
                        f"phi: {row.get('phi', '')}",
                        f"sw: {row.get('sw', '')}"
                    ]),
                    showlegend=False
                ))
        
            fig_pay.update_layout(
                height=1000,
                margin=dict(l=70, r=50, t=65, b=95), #####
                plot_bgcolor='white',
                paper_bgcolor='white',
                dragmode=False
            )
            
            fig_pay.update_yaxes(
                title_text="Depth (m)",
                autorange='reversed',
                range=depth_range,
                showgrid=True,
                gridcolor='lightgray',
                ticks="outside",
                showline=True,
                linewidth=2,
                linecolor='black',
                mirror='all',  # Show all borders (top, bottom, left, right)
                tickfont=dict(size=12)
            )
            
            fig_pay.update_xaxes(
                range=[0, 1],
                showline=True,
                linewidth=2,
                linecolor='black',
                mirror='all',  # top and bottom borders
                showticklabels=False,
                ticks='',
                showgrid=False
            )
            
            # Add white rectangle to cover right border line
            fig_pay.add_shape(
                type="rect",
                xref="paper", yref="paper",
                x0=1, x1=1.02,    # Slightly outside the right border
                y0=0, y1=1,
                fillcolor="white",
                line_width=0,
                layer="above"
            )

            st.plotly_chart(fig_pay, use_container_width=True, config={
                'displayModeBar': True,
                'modeBarButtonsToRemove': [
                    'select2d', 'lasso2d', 'resetScale2d', 'autoScale2d'
                ],  # Removed 'toImage' to keep save PNG button
                'displaylogo': False,
                'staticPlot': False,
                'responsive': True
            })



        # Phi/Sw plots
        with col2:
            st.markdown("#### Porosity / Water Saturation")
        
            phi_range = [
                min(selected_df['phi'].min(), selected_df['phi_predicted'].min()) - 0.03,
                max(selected_df['phi'].max(), selected_df['phi_predicted'].max()) + 0.03
            ]
            sw_range = [
                min(selected_df['sw'].min(), selected_df['sw_predicted'].min()) - 0.05,
                max(selected_df['sw'].max(), selected_df['sw_predicted'].max()) + 0.05
            ]
        
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go
        
            # Create two subplots side by side
            fig = make_subplots(rows=1, cols=2)
        
            # Porosity traces
            fig.add_trace(go.Scatter(
                x=selected_df['phi'], y=selected_df['DEPT'],
                mode='lines', name='Phi Actual', line=dict(color='blue', width=2)
            ), row=1, col=1)
        
            fig.add_trace(go.Scatter(
                x=selected_df['phi_predicted'], y=selected_df['DEPT'],
                mode='lines', name='Phi Predicted', line=dict(color='red', width=2, dash='dot')
            ), row=1, col=1)
        
            # Sw traces
            fig.add_trace(go.Scatter(
                x=selected_df['sw'], y=selected_df['DEPT'],
                mode='lines', name='Sw Actual', line=dict(color='green', width=2)
            ), row=1, col=2)
        
            fig.add_trace(go.Scatter(
                x=selected_df['sw_predicted'], y=selected_df['DEPT'],
                mode='lines', name='Sw Predicted', line=dict(color='orange', width=2, dash='dot')
            ), row=1, col=2)
        
            # Update x-axes
            fig.update_xaxes(
                title_text="Porosity",
                title_standoff=20,   # Increase space between axis line and title
                range=phi_range,
                showgrid=True,
                gridcolor='lightgray',
                # overlaying='x',      # overlay on bottom x-axis 1
                # side='top',           # place ticks & labels on top
                # showticklabels=False,  # show numbering on top
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
                title_standoff=20,   # Same here
                range=sw_range,
                showgrid=True,
                gridcolor='lightgray',
                # overlaying='x',      # overlay on bottom x-axis 1
                # side='top',           # place ticks & labels on top
                # showticklabels=False,  # show numbering on top
                ticks="outside",
                showline=True,
                linewidth=2,
                linecolor='black',
                mirror='ticks',
                tickfont=dict(size=12),
                row=1, col=2
            )
            
            # Also update layout margin bottom to ensure space
            fig.update_layout(
                margin=dict(l=50, r=50, t=80, b=100),  # increase bottom margin to 100
            )


        
            # Update y-axes
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
                # title_text="Depth (m)",      # Add title if desired
                autorange='reversed',
                showgrid=True,
                gridcolor='lightgray',
                ticks="outside",
                showline=True,
                linewidth=2,
                linecolor='black',
                tickfont=dict(size=12),
                showticklabels=True,         # Show labels on y-axis
                row=1, col=2
            )

        
            # Layout and legend on top (unchanged)
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
                # Removed all shapes (no black border rectangles)
            )
        
            # Bottom static legend (keep exactly as before)
            fig.add_annotation(
                text=(
                    "<span style='color:blue'><b>───</b></span> Phi Actual&nbsp;&nbsp;"
                    "<span style='color:red'><b>- - - - -</b></span> Phi Predicted&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
                    "                                    "
                    "<span style='color:green'><b>───</b></span> Sw Actual&nbsp;&nbsp;"
                    "<span style='color:orange'><b>- - - - -</b></span> Sw Predicted"
                ),
                xref="paper", yref="paper",
                x=0.5, y=-0.10,
                showarrow=False,
                font=dict(size=13, color="black"),
                align="center",
                bgcolor="rgba(255,255,255,0.8)",
                # bordercolor="black",
                borderwidth=1
            )
        
            st.plotly_chart(fig, use_container_width=True, config={
                'displayModeBar': True,
                'modeBarButtonsToRemove': [
                    'select2d', 'lasso2d', 'resetScale2d', 'autoScale2d'
                ],  # Removed 'toImage' to keep save PNG button
                'displaylogo': False,
                'staticPlot': False,
                'responsive': True
            })




    else:
        st.error("Missing required columns in the selected well file.")
else:
    st.info("Upload well CSV files from the sidebar to begin analysis")
