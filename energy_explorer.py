import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Energy Data Explorer",
    page_icon="?",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stPlotlyChart {
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(file_path=None):
    """Load and prepare the dataset"""
    if file_path:
        df = pd.read_excel(file_path)
    else:
        # Sample data structure based on the Excel file
        data = {
            'year': [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
            'IMI': [None, None, None, None, None, None, 26.51, 21.26, 23.92, 23.24, 25.91],
            'Linetal': [49.93, None, 41.99, 32.20, 33.75, 29.68, 26.34, 19.85, 47.85, 26.81, 31.98],
            'oil_million_bbls': [35.00, 40.91, 37.14, 30.53, 34.44, 37.12, 36.93, 31.00, 35.77, 45.33, 51.41],
            'gas_million_bbleq': [78.39, 75.00, 69.50, 60.88, 52.53, 49.30, 45.50, 40.43, 40.01, 43.42, 40.00],
            'energy_million_bbleq': [113.39, 115.92, 106.64, 91.41, 86.97, 86.42, 82.43, 71.43, 75.79, 88.75, 91.40],
            'bbleq_ratio': [0.45, 0.55, 0.53, 0.50, 0.66, 0.75, 0.81, 0.77, 0.89, 1.04, 1.29]
        }
        df = pd.DataFrame(data)
    
    # Clean column names
    df.columns = df.columns.str.replace(' ', '_').str.replace(':', '_')
    
    return df

def create_overview_metrics(df):
    """Create overview metrics"""
    latest_year = df['year'].max()
    earliest_year = df['year'].min()
    
    # Calculate key metrics
    total_years = latest_year - earliest_year + 1
    avg_oil_production = df['oil_million_bbls'].mean()
    avg_gas_production = df['gas_million_bbleq'].mean()
    avg_total_energy = df['energy_million_bbleq'].mean()
    
    # Calculate growth rates
    oil_growth = ((df['oil_million_bbls'].iloc[-1] / df['oil_million_bbls'].iloc[0]) - 1) * 100
    gas_decline = ((df['gas_million_bbleq'].iloc[-1] / df['gas_million_bbleq'].iloc[0]) - 1) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            label="?? Years of Data",
            value=f"{total_years}",
            delta=f"{earliest_year}-{latest_year}"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            label="??? Avg Oil Production",
            value=f"{avg_oil_production:.1f}M bbls",
            delta=f"{oil_growth:+.1f}% total growth"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            label="?? Avg Gas Production",
            value=f"{avg_gas_production:.1f}M bbleq",
            delta=f"{gas_decline:+.1f}% total change"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            label="? Avg Total Energy",
            value=f"{avg_total_energy:.1f}M bbleq",
            delta="Combined O&G"
        )
        st.markdown('</div>', unsafe_allow_html=True)

def create_time_series_plot(df, selected_metrics):
    """Create interactive time series plot"""
    fig = make_subplots(
        rows=len(selected_metrics), cols=1,
        subplot_titles=selected_metrics,
        vertical_spacing=0.08,
        shared_xaxes=True
    )
    
    colors = px.colors.qualitative.Set1
    
    for i, metric in enumerate(selected_metrics):
        if metric in df.columns:
            # Remove null values for plotting
            mask = df[metric].notna()
            x_data = df.loc[mask, 'year']
            y_data = df.loc[mask, metric]
            
            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode='lines+markers',
                    name=metric.replace('_', ' ').title(),
                    line=dict(color=colors[i % len(colors)], width=3),
                    marker=dict(size=8),
                    hovertemplate=f'<b>{metric.replace("_", " ").title()}</b><br>' +
                                'Year: %{x}<br>' +
                                'Value: %{y:.2f}<br>' +
                                '<extra></extra>'
                ),
                row=i+1, col=1
            )
    
    fig.update_layout(
        height=200 * len(selected_metrics) + 100,
        title_text="?? Time Series Analysis",
        title_x=0.5,
        showlegend=False,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Year", row=len(selected_metrics), col=1)
    
    return fig

def create_correlation_heatmap(df):
    """Create correlation heatmap"""
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'year']
    
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            color_continuous_scale='RdBu_r',
            aspect='auto',
            title="?? Correlation Matrix"
        )
        
        # Add correlation values as text
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                fig.add_annotation(
                    x=i, y=j,
                    text=str(round(corr_matrix.iloc[j, i], 2)),
                    showarrow=False,
                    font=dict(color="white" if abs(corr_matrix.iloc[j, i]) > 0.5 else "black")
                )
        
        fig.update_layout(height=500)
        return fig
    return None

def create_distribution_plot(df, selected_metric):
    """Create distribution plot for selected metric"""
    if selected_metric in df.columns:
        data = df[selected_metric].dropna()
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Distribution', 'Box Plot'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=data,
                nbinsx=10,
                name="Distribution",
                marker_color='lightblue',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(
                y=data,
                name="Box Plot",
                marker_color='lightgreen'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text=f"?? Distribution Analysis: {selected_metric.replace('_', ' ').title()}",
            showlegend=False,
            height=400
        )
        
        return fig
    return None

def create_comparison_plot(df):
    """Create comparison plot between oil and gas production"""
    fig = go.Figure()
    
    # Oil production
    fig.add_trace(go.Scatter(
        x=df['year'],
        y=df['oil_million_bbls'],
        mode='lines+markers',
        name='Oil Production',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    
    # Gas production
    fig.add_trace(go.Scatter(
        x=df['year'],
        y=df['gas_million_bbleq'],
        mode='lines+markers',
        name='Gas Production',
        line=dict(color='#ff7f0e', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="?? Oil vs Gas Production Comparison",
        xaxis_title="Year",
        yaxis_title="Production (Million Barrels/Equivalent)",
        hovermode='x unified',
        height=500
    )
    
    return fig

def main():
    """Main application"""
    # Header
    st.markdown('<h1 class="main-header">? Energy Data Explorer</h1>', unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader("Upload your Excel file", type=['xlsx', 'xls'])
    
    # Load data
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.success("? Data loaded successfully!")
    else:
        df = load_data()
        st.info("?? Using sample data. Upload your Excel file to explore your own data.")
    
    # Sidebar controls
    st.sidebar.header("??? Controls")
    
    # Data overview
    st.sidebar.subheader("?? Data Overview")
    st.sidebar.write(f"**Records:** {len(df)}")
    st.sidebar.write(f"**Time Period:** {df['year'].min()}-{df['year'].max()}")
    st.sidebar.write(f"**Columns:** {len(df.columns)}")
    
    # Filter controls
    st.sidebar.subheader("?? Filters")
    
    # Year range selector
    year_range = st.sidebar.slider(
        "Select Year Range",
        min_value=int(df['year'].min()),
        max_value=int(df['year'].max()),
        value=(int(df['year'].min()), int(df['year'].max()))
    )
    
    # Filter data by year range
    filtered_df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
    
    # Metric selection
    numeric_columns = [col for col in df.columns if col != 'year' and df[col].dtype in ['float64', 'int64']]
    selected_metrics = st.sidebar.multiselect(
        "Select Metrics for Time Series",
        options=numeric_columns,
        default=numeric_columns[:3] if len(numeric_columns) >= 3 else numeric_columns
    )
    
    # Main content
    if not filtered_df.empty:
        # Overview metrics
        st.subheader("?? Key Metrics Overview")
        create_overview_metrics(filtered_df)
        
        # Main visualizations
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Time series plot
            if selected_metrics:
                st.subheader("?? Time Series Analysis")
                fig_ts = create_time_series_plot(filtered_df, selected_metrics)
                st.plotly_chart(fig_ts, use_container_width=True)
            
            # Comparison plot
            if 'oil_million_bbls' in filtered_df.columns and 'gas_million_bbleq' in filtered_df.columns:
                st.subheader("?? Oil vs Gas Production")
                fig_comp = create_comparison_plot(filtered_df)
                st.plotly_chart(fig_comp, use_container_width=True)
        
        with col2:
            # Distribution analysis
            st.subheader("?? Distribution Analysis")
            dist_metric = st.selectbox(
                "Select metric for distribution:",
                options=numeric_columns
            )
            
            if dist_metric:
                fig_dist = create_distribution_plot(filtered_df, dist_metric)
                if fig_dist:
                    st.plotly_chart(fig_dist, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("?? Correlation Analysis")
        fig_corr = create_correlation_heatmap(filtered_df)
        if fig_corr:
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # Data table
        st.subheader("?? Data Table")
        show_raw_data = st.checkbox("Show raw data")
        if show_raw_data:
            st.dataframe(filtered_df, use_container_width=True)
        
        # Download processed data
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="?? Download Filtered Data as CSV",
            data=csv,
            file_name=f"energy_data_{year_range[0]}_{year_range[1]}.csv",
            mime="text/csv"
        )
        
        # Statistical summary
        with st.expander("?? Statistical Summary"):
            st.write(filtered_df.describe())
    
    else:
        st.warning("?? No data available for the selected filters.")
    
    # Footer
    st.markdown("---")
    st.markdown("**Energy Data Explorer** | Built with Streamlit and Plotly")

if __name__ == "__main__":
    main()